# EMBER findings

本文保存跨轮发现及其适用边界，不记录执行授权。最新结果见§241；当前授权和接手位置见[progress](progress.md)。
前部的专题复核与后部编号结论按各自日期解释，历史待办不构成自动实验计划。

## O1200/C600 既有执行轨迹：失败定位材料的严格边界（2026-09-22）

专家要求从已完成 D1 `CC` 轨迹观察最早行为差异，而非继续训练或扩展消融。事后按每 suite 的 task/state 升序固定选择
一个旧成功/新失败、teacher 条件相同的行：Spatial 5/1、Object 2/0、Goal 0/1、Long 4/1；因选择使用已有结果，四对
不构成新的性能率样本。所有原件是 compact capture，保存 action chunk 和 replan state 而不含逐时刻 RGB；将保存的
实际执行前缀重放后，8/8 trajectory 的 replan Pi05 state（最大绝对误差0）、success/steps、stage predicate
transitions、ever/final/peak 均与原始正式行一致。因此远程
[行为复核包](docs/review_materials/20260922/writer_causal_diagnostics/behavior_pairs/README.md)中的视频、action/state 和
predicate 时间线是已验证的保存动作重放，不是新 policy rollout。它没有自行判定“抓取/目标/摆动”等错误类别，也不能证明
唯一根因或改变现有训练结论；该判定留给专家。

## Writer 因果路径与辅助梯度诊断：完整证据边界（2026-09-22）

冻结 D1/D2/D3 已完整结束（544条路径 probe、112条路径闭环、16条虚拟更新、54条微训练更新、192条终点 probe、
48条终点闭环），完整报告与全部结构化原始行已作为远程
[专家审计包](docs/review_materials/20260922/writer_causal_diagnostics/README.md)上传；canonical 本地 study 仍在
`/data0/user/ymdai/ember_runs/writer_causal_diagnostics_20260922/`。它不使用 Validation/Test，也不进入 checkpoint 选择。

D1 显示只有 Procedure donor replacement（CW）即可在 O1200/C600 上比只有 Core donor replacement（WC）产生明显更大的
动作和 fused/LoRA 变化；二者都通过真实解码器消费者。C600 小面板 CC/CO/CW/WC/WW=2/5/3/5/4，O1200 CC/WW均为10/16，
所以这不是正确视频内容或顺序在闭环上必要的证据。

D2 的生产 native BF16 复放中，辅助项定义为已含权重的 `g_A=g_J-g_Q`。四个资产/窗口的 `||g_A||/||g_Q||` 是
.978/.728/.890/.526，全局 `cos(g_Q,g_A)` 是 .113/.192/.243/.286，且各 J preclip norm<1；因此当前辅助项既非零也非
主项的标量复制，但不能由梯度量级推出闭环因果效力或将其分解为 `tau=1`、前5步和`1/3`各自的效应。

D3 固定 C600、events601..618、同一LR的18步端点中，J=6/16 breadth4、Q=5/16 breadth5、M=5/16 breadth4；三臂都有
相对初始2/16的新增成功且 success set 不同。J仅在总成功多一条而没有覆盖/保持上的统一优势，故不选择 J/Q/M，不修改
正式 cross-episode 辅助配对路线，也不声称辅助项整体有效或无效。

本轮中间工程曾修正两项问题：D2以生产联合LoRA余切的一次Writer VJP为准，辅助梯度定义为`g_A=g_J-g_Q`；
分组CSV同名列覆盖只影响汇总，不改变训练/闭环。旧错误输出作为工程原件封存，以最终审计包为准，不再保留“尚无结果”的过期状态。

本文件保留跨轮结论及其适用边界；历史段落的“当前／下一步／active”只表示当时时点。
当前Owner要求见[稳定要求](docs/current_owner_requirements.md)，当前授权、设计和执行状态只看[progress](progress.md)。
完整历史索引、旧设计及原始证据入口见[research_history](docs/research_history.md)。

## Writer 动作监督与辅助目标的证据边界（2026-09-22）

当前主监督使用π0.5原生flow matching：随机flow time、完整50步动作chunk、真实7维速度MSE，并让视频生成的
同一套LoRA在同任务另一episode上接受检验。这一基础目标有原生模型和实现依据。训练时单次随机时间forward
已经构成完整FM监督，不要求反传10步推理解算器。

额外教学项固定`tau=1`、只监督前5步、权重`1/3`。它有强调纯噪声端点和实际执行前缀的动机，尚缺少独立
消融证明。两项分别求均值后相加，名义上前5步占组合目标32.5%，每个前5步位置的系数是其余位置的4.33倍，
总系数25%来自精确`tau=1`；这些是公式权重，不是实测梯度占比。端点最优预测对应条件平均动作，也不能直接
等同于10步flow解算后的完整动作。

历史24任务same-video辅助与cross-episode辅助都保留端点、前缀和`1/3`，因此只识别辅助episode配对；它们
无法回答额外教学项相对纯跨episode完整FM是否有净收益。已合入`main`的PR #3在当前36任务上完成一次fresh匹配
对照，真实事件preflight、所有400行、manifest和worker完成均通过：cross-episode correct400在200..1200为
151/120/154/137/129/124，对原same-video的110/92/115/87/117/80逐点为+41/+28/+39/+50/+12/+44。这个方向一致的
同合同改善支持“改变**辅助episode配对**影响学习”，不能被表述为采用标准VLA损失后的修复。

它尚非稳定的总体恢复：唯一最高600=154，比原Writer最高117高37，但比冻结MT-BC300=155低1；200→400、
400→600、600→800的R/G/L=84/36/67、80/74/40、101/36/53。600相对same-video同节点的R/G/L=78/76/37、
J=.4084；Spatial净+45，而Goal/Object各净−8/−7。早停在1200以持续下降触发，不能因所有同节点正差或单一154
峰值把能力交换写成相邻保持。

预登记的154分支完成same-task-other=160与cross-suite-wrong=153；correct154相对wrong只+1，且other高6，
不支持正确视频内容恢复。由于分数未达156，shuffle/reverse未执行，顺序特异性未测。纯主FM仍是附加假设更少的
独立基准，若执行需单独预注册，不能与PR #3共享单变量归因；本结果也不证明`tau=1`、前5步或`1/3`任何一项必要。

## 当前覆盖重训 Writer 的视频证据（2026-09-21）

新24/8/8＋12协议的唯一Writer选中step1000，correct/other/wrong/shuffled/reversed完整Validation400依次为117/119/107/86/62。
correct相对wrong只高10/400；以8个task为簇、保留每task全部50配对初态的20,000次bootstrap，wrong−correct为−2.5pp，95%百分位区间[−6.5,+0.5]pp。差异主要来自Object1和Spatial3，不能清楚证明正确视频内容相对跨suite错误视频是必要增量。
reversed−correct为−13.75pp，区间[−24.75,−3.5]pp，支持该冻结模型对时间方向敏感；shuffled−correct为−7.75pp，区间[−16.5,+0.51]pp，点估计下降但任务簇区间跨零。other−correct为+0.5pp，区间[−2,+3.25]pp，没有严重同任务换视频退化。
三种干预各400行、全部worker正常退出；wrong的8个donor均跨suite，shuffled/reversed每条都重排真实RGB后完整forward，且与correct使用同一task/state/teacher序号与RNG合同。现阶段把内容特异性记为科学上不清楚，暂停新Test及后继FT/RL/外部比较，等待Owner裁决；MT-BC独立训练随后按原合同完成。完整图表与原件索引见[覆盖重训报告](docs/review_materials/20260921/coverage_retraining/report.md)。

MT-BC后来按原合同完成10个完整correct400节点，step500持续下降早停，唯一最高step300为155/400；Writer唯一最高step1000为117/400。两模型在同一Validation8逐行配对，Writer−MT-BC为−38/400，任务簇bootstrap95%百分位区间[−29.75,+6.5]pp，成功任务覆盖4/8对6/8。Writer在Long1/Goal6/Object1局部更好，主要损失来自Spatial3、Object6和Spatial6。两方法训练演示范围与每更新任务/查询语义不同，不能把此差异归为单一架构因果；也不能用Validation差值冒充尚未执行的新Test硬门槛。MT-BC完成后上述视频内容不清楚的暂停裁决仍有效，Test/FT/RL/外部比较待Owner明确决定。

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

当前最有支持的工作解释是条件表示/参数共享与正样本FM耦合，继续适配训练任务时未形成或保留广泛迁移；训练组织/池/优化差异只是可能放大因素。具体首次失效模块、视频因果贡献和动作失败阶段仍不可由现有证据唯一识别。完整证据、反证、覆盖范围及原因排序见[审计报告](docs/analyses/horizon_k1_evidence_review_20260909.md)。24-task草稿与所有实验保持暂停。

## 23. 头部共享和路由限制有近等价历史，先分析实际功能再改模（2026-09-09）

当前D约占Writer89.38%，不据此认定过拟合。旧Target-Owned已经使用同target内跨rank共享native读出，20,594,688末投影参数，正式99/76/86/68；当前把329,515,008独立D收为同样大小的rank共享D并非未试过的新机制。旧前端与配方不同，不能单独否定当前受控对照，但它应降低无新机制证据就重训的优先级。

旧v5.2/v6已有纯language只Q、video content作Value/残差的真实实现；Dynamic-K Semantic-Address、DirectFamilyB、task-grounded D/G也表明这种限制并不充分。删除当前裸语言残差只能排除该条可达路径，不能宣称视频动态必要性或闭环根因已证实，更不能扩大为删除全部视频条件共同语义Value。

Owner已授权直接做冻结分析实验与架构内部拆解，先分析后决定正式修改/训练。应同时补实际行为阶段与正常输入下的功能敏感性；分量大小、局部梯度、目标谓词或小面板分别保留适用边界。完整历史与来源见research_history §2补充，当前固定诊断口径见active design §8.2.4。


## 24. 冻结接口依赖与实际失败阶段已分开，仍不能唯一识别泛化根因（2026-09-09）

原200/600在相同train24×teacher46×16queries上，P4/cross实际非零且单cross lesion会改变真实FM预测。首层language残差移除FM增加.026782/.028808，23/24与24/24任务变差；只移除首层language检索增加.000188/减少.000107、方向混合。当前训练侧功能对首层直接language内容依赖较强，但后续路由/跨层补偿没有被固定；不是全局路由无用、语言捷径或fresh删除有效的证明。

预登记validation8×四state×两checkpoint回放完成，9/32→7/32；实际配对通过，但与各自历史7/32成功集合不完全相同，因此图像只解释本次轨迹，原正式分数不改。所见Spatial错误对象/实例、BBQ200正确目标到600干扰物、Long独立子目标与混合抓取/干扰，以及实际成功双目标组合，均不能统一为motor失效或单个过程层故障。

新证据提高了对compiler直接条件内容路径做单变量fresh检验的信息价值；该对照保留首次language检索、原前端/decoder/4×64，不同时改D共享或task组织。它是检验归纳偏置的学习实验，成功/失败必须由strict400与训练侧行为裁决，不将冻结activation lesion的OOD损失变化当作预先答案。完整原件、范围和边界见[冻结诊断报告](docs/analyses/horizon_k1_frozen_diagnostics_20260909.md)。

## 25. 首层直接语言内容移除尚未成为整体修复（2026-09-09）

完整单变量fresh首段100/200 correct75/110对原55/110；新200和原200总分/breadth相同，保留86、新增24、丢24。新100→200仍净增35、四suite净增，但breadth6/8不变、global1/23持续0，churn75/J=.4231。新200 train96=46、breadth18，低于原52/breadth20；训练侧S/O/G/L为12/15/16/9→11/12/14/9，三个suite下降、Long持平。同3072query held FM均值.111184/.111353接近并未对应训练行为等价。

因此不能把原裸language残差认定为唯一泛化根因，也不能称删除后修复。新模型仍有真实学习获取，首段未覆盖原300/400由110回落86/87的区间；同变量固定续至300/400是保持检验，不是新的架构配方或按loss无界续训。低分只约束实测阶段和接口，后续必须用完整保持区间与训练侧证据裁决。完整表、配对与范围见[首段报告](docs/analyses/horizon_k1_first_query_only_20260909.md)。

## 26. 语言内容移除的局部收益不能替代任务保持与迁移（2026-09-09）

新100/200/300/400 correct75/110/106/103，对原55/110/86/87。后两节点有20/16个净成功优势，但自身200后不增长；BBQ28→24→10的回落仍在，cream cheese等增益掩盖部分损失。300→400 R/G/L77/26/29、churn55、J=.5833；400breadth6、Long9，两个task全程0。Spatial任务3在200/400均3/50却成功集合无交集，Long双物9→8只保留1，说明近似总分不等于稳定能力。

新400 held-video train96=59、breadth22，对新20046/breadth18为35/24/11；与原40059/breadth21为46/13/13。新400 S/O/G/L12/21/19/7，对原17/18/17/7，Spatial少5由Object/Goal增益抵消；训练获取真实，不能解释为全局未学会。配对3072query held FM新400.105738、原.106272，只有11/24任务新更低，内部均值接近不代表行为等价。

本轮否定该处内容移除足以修复当前缺口，保留局部贡献与其它接口的不确定性。不自动沿用已重现的“训练能力继续提高、目标能力不扩展”趋势续500/600，也不把它误写成训练完全饱和。下一步须有新的可区分机制和近等价历史边界；§23关于Target-Owned rank共享已有失败的事实仍有效。完整报告[首层内容对照至400步](docs/analyses/horizon_k1_first_query_only_20260909.md)，本段原件`k1_first_query_only/segment200_400/round_evidence.json`。

## 27. 训练内功能对应的增强并未带来验证迁移（2026-09-09）

按active design§8.2.7，新200/400、teacher46/47的既有train24 LoRA在固定32query上交叉执行，含source共74,496query预测，全部无梯度且实际time/noise配对。400的24task在两teacher、两个16-query半面板均比其余23task adapter均值低FM；跨task margin由.009396增至.015491，自身FM由.111621降至.106557。与train96 46→59、validation110→103并列，进一步降低“训练内条件功能普遍未形成”的优先级。

同suite margin从.003599到.005578，但Object及部分Long细分差值小，400仅18/24task满足两teacher、两半面板均优于同suite均值。policy本身读正确language，合理共享通用技能无需每task独有LoRA；本矩阵既不单独定位错误路由，也不证明FM足以支持闭环。数值缓存核对约.9%–1.1%逐点loss相对差，不把细小排名当精确机制证据。

当前global语言读出有位置而非无序mean，但其静态token到单query attention尚无预训练上下文task-token语义；现有Z视觉重读已经能读取这些上下文tokens，不能说语义缺失。若后继检验更直接的上下文过程条件，只能作为同源信息访问的受控变量，保留四组动作关系/视觉核实主图；不能据结构名字宣称修复。完整报告[冻结功能对应](docs/analyses/horizon_k1_functional_assignment_20260909.md)，原件`k1_first_query_only/functional_assignment/`。

## 28. 上下文过程条件首段未显示整体优势，早期恢复仍需与保持分开（2026-09-09）

只替换四组过程条件为同次冻结Gemma的逐帧exact task-token读取，保留reader参数、first-query-only、完整过程图/native D及全部实际采样。fresh100/200 correct52/103，前轮75/110；200 train96为41，前轮46。全部完整checkpoint、896个新bank条件、曝光和三面板配对通过，均exit0；没有发现工程失败足以解释结果。

200 S/O/G/L1/54/37/11、breadth7，但Spatial1与Goal23仅各1/50；100/200的Goal23成功state不同，Long4→11也只保留1次。前轮200→本轮200总R/G/L76/27/34、Long2/9/8，不把Long总数或breadth当稳定能力。train96为35/6/11，breadth18保持，局部双moka新增2被book丢3等抵消。held3072query均值.111031/.111184接近仍不能推出行为等价。

相对早期缺口23→7、自身52→103支持继续区分较慢获取与后续保持，不能说明当前改动优于前轮。是否值得投入一个固定300/400区间与是否已经有收益是不同判断；后续若无实质获取/保持优势，不按loss无限延长。负结果只约束本次表示来源及共享reader组合，不独自否定上下文语义、整个过程图或纯FM。完整报告[上下文条件首段](docs/analyses/horizon_k1_frame_contextual_20260909.md)。

## 29. 上下文条件完整保持段未修复获取/保持，FM改善不能替代行为（2026-09-09）

本轮100/200/300/400 correct52/103/79/90，前轮75/110/106/103；400 train96为49，前轮59。相邻300→400 R/G/L63/27/16，200→40056/34/47；BBQ25→3→1且200原25成功到400全部丢失。400 global1/3各1、23/32为0，四suite非零不等于稳定广度。完整source/视频/state/RNG与checkpoint配对通过，没有工程失败证据解释整体负结果。

400 held FM .105074533低于本轮200全部24task，且均值低于前轮400 .105737594；训练面板41→49仍有获取，不能将问题全称为训练任务普遍退化。平均拟合、更丰富上下文访问与实际可保持闭环能力之间仍有缺口；本次干预无整体收益，结束原样续训。该结论只限制实际检验的条件来源/共享reader组合，不单独否定完整H/过程图/原生语言，也不授权新正式方法。

完整[100–400报告](docs/analyses/horizon_k1_frame_contextual_20260909.md)与`k1_frame_contextual/segment200_400/round_evidence.json`保存原件。Owner目前仅授权原因诊断，实际状态以progress为准。


## 30. 完整因果诊断：功能拟合、接口获取和后续执行必须分开（2026-09-09）

B2完整native FM和10-step动作MSE仍无法排序闭环；context400比first-query400前5动作MSE在19/24 task更低，train/validation却49/90对59/103。padding拟合不能解释大部分改善，夹爪事件覆盖存在，小面板事件退化未稳健复现。

B3全24task×四state×五臂480行完整：normal51，H-read/Compiler/visual零54/52/54，分别R/G/L47/7/4、40/12/11、45/9/6；全后端语言零40、33/7/18。当前单支路贡献混合，联合语言依赖存在；不是fresh删除或视频必要性证明。

八task局部final64的C闭环normal/P4/C/AB/expert18/17/20/15/16（各32）；task7 normal/P4均1，C/AB均4，说明当前冻结D有局部有效策略可达，但不能区分P4信息与Compiler/优化。C净增伴随Goal/Long损失。直接A/B fit .095526→.039659，独立 .101330→.116502；换time/noise后原fit收益只剩6.8%。固定随机点局部求解不是容量上界，也不证明正式训练复用了噪声（1600condition各自seed不同）。

Long36/38固定state32回放均已转向第二对象后未完成操作；旧专家同状态也失败。不能把这些例子统称未选择下一目标或Writer独有问题，也不能把专家658/1200当每个任务的能力保证。后续优先验证有效功能修正和实际到达/恢复状态上的行为，再分别检验语言简化与P4→C获取；不按D参数占比、局部loss或任务数量猜测直接大改。

完整[诊断报告](docs/analyses/horizon_k1_causal_diagnostics_20260909.md)与`causal_diagnostics_20260909/summary.json`保留全部原件；新诊断闭环676行完整，正式架构/训练未改、未启动正式训练，无held梯度或最终视频controls。


## 31. 因果解释纠正与探索授权澄清（2026-09-09）

Owner指出并确认：冻结置零不能替代删路径后fresh学习，局部decoder可达不能证明整个架构易于学出正确表示；旧强v5.2/v6的同类FM能力要求实际解释架构/配方差距。§30及诊断报告中的语言“非统一解释”不能作为排除主要原因的结论；现有实测数值保留，尚未完成根因与修正验证。

Owner允许为探索修改架构和训练方式并进行实验性训练，限制在正式采纳候选并启动下一轮正式训练之前汇报。本次重新设立独立原因分析goal，具体计划见`docs/designs/horizon_causal_learning_plan_20260909.md`，当前状态以progress顶部为准。

## 32. 验证下降集中于反复丢失BBQ，实例呈目标选择变化（2026-09-09）

原始/first-query/contextual三版200→400的train96分别52→59、46→59、41→49；validation110→87、110→103、103→90。BBQ分别24→3、28→10、25→1，扣除它后其它七task总成功86→84、82→93、78→89。因此不能将总分下降概括为所有任务同步遗忘；持续弱的Spatial/Goal23/Long32获取问题与BBQ保持问题需要分别解释。全部原contract、teacher/state/RNG实际配对复算保留于`causal_learning_20260909/existing_learning_and_retention.json`。

当前contextual200/400在固定states0/12/25/37的8条正常correct回放全部完成，200为3/4、400为0/4，8条均复现历史成功/失败，终态BDDL谓词一致。双相机轨迹显示200四例均操作正确BBQ（state0过晚运输而超时）；400四例均转向绿色干扰瓶，其中state25/37抓起并运到篮子区域，正确BBQ留在桌面。这些实例支持目标选择变化，反对把它们一概解释为抓取/运输能力消失；无法单独确定语言、过程表示或生成LoRA哪层导致该变化。

task因重复下降事后选定、states沿用旧诊断固定集合，属于描述性实例，不能作全局率、checkpoint选择或根因识别。原件、逐例解释及限制在`causal_learning_20260909/retention_replay/completed_summary.json`，完整索引见[当前因果计划](docs/designs/horizon_causal_learning_plan_20260909.md)。

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

采用局部无变化参照作为下一受控候选，保留C其余结构和普通FM；须经真实profile和fresh后续节点才能判断。完整论证、有限面板和路径替换边界见[机制复核§9–10](docs/analyses/video_mechanism_reassessment.md)。1472条新闭环及逐行配对、逐task/suite、success-set和task bootstrap在runs/analysis/video_mechanism_20260911/{behavior_summary,functional_comparison}.json及paths/behavior_summary.json。现有validation C43/50对source47未建立可信稳定source收益，EMBER目标与本次goal均未达成。


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
  `docs/analyses/horizon_k1_evidence_review_20260909.md`§4.1。支持架构与学习组织交互，不支持任一未识别配方因素自动成为修复。
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

下一项机制分析见[过程获取提案](docs/analyses/video_process_acquisition_analysis.md)。源码复核纠正一种容易混同的历史：
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
检查未读取action数值、held数据或运行环境。具体生产来源、数据revision及数值见[提案§9](docs/analyses/video_process_acquisition_analysis.md)。

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

[Execution-Aligned设计](docs/designs/execution_aligned_writer_design.md)据实际post-action时序，将唯一Writer的主FM与动作留出
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

条件性[冻结正例审计§6](docs/analyses/frozen_positive_replication_audit.md#6-条件性正确收益复核合同新outcome产生前登记)
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

按新outcome前登记的[复核合同§6](docs/analyses/frozen_positive_replication_audit.md#6-条件性正确收益复核合同新outcome产生前登记)，
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

完整推导、竞争预测及停止分支见[视频信息与可识别性](docs/analyses/video_information_identifiability.md)。
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

按新增outcome前登记的[冻结生成诊断](docs/analyses/frozen_local_action_decode_audit.md)，固定原local两臂step200，
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

源码与原始文献审查见[可识别性分析§7](docs/analyses/video_information_identifiability.md#7-显式运动对应能补什么以及为什么尚不足以启动新writer)。
当前任务token视觉读取／native H端点读取没有显式同物理点约束；不因此否认其隐式运动知识。
跟踪可见表面与生成具体7维机器人动作是不同问题，旧局部动作头失败不直接否决前者。

Im2Flow2Act等正证据同时依赖目标物体绑定、执行状态对应或动作重定向，不能只抽取“flow有效”来支持新Writer。
固定运动先验仍是RGB的函数；点对应能改善可用表示，不自动解决跨演示初态的操作关系及唯一LoRA的行为传递。
仅评估光度warp或展示轨迹图无法区分获取不足与编译不足，也可能只反映背景／机器人／相机运动。

当前不采用直接换成／追加dense-flow编码器、其余跨episode FM与Compiler不变的提案；不启动只验证跟踪后
必然进入Writer训练的probe链。目标物体运动仍保留为候选，重提时须给出可改变决策的跨初态功能判别，
并区分感知失败、任务相关性不足与编译传递失败。尚无新实验design、权重下载、forward或闭环结果；goal未完成。

## 77. 冻结source在state-free输入下已有动作生成能力，补state的预登记前提未通过（2026-09-13）

[预注册输入诊断](docs/analyses/source_state_input_audit.md)固定train24×16位置，逐臂8个配对噪声、10步原生采样；
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

[预登记端点×视角诊断](docs/analyses/source_endpoint_readout_audit.md)先确认当前execution-aligned observer为agentview，
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

[Native双相机设计](docs/designs/native_dual_video_writer_design.md)的两臂fresh200及8面板全部完成，1,984行，
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

[可识别性分析§9](docs/analyses/video_information_identifiability.md#9-跨初态操作关系新增监督必须区别于旧状态条件化与任务记忆)
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

[登记诊断](docs/analyses/cross_init_relation_retrieval_audit.md)已完成：train24、action16–19单演示与诊断42–45交叉，
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

[操作语义可行性](docs/analyses/operation_semantics_feasibility.md)完成train24固定192episode、6,668状态位置的CPU恢复，
源7bd8a6f5、84.34秒exit0。两指同对象接触共2,878对象帧、493次转换，186/192episode存在转换。
这证明所选物理事实不是task常数；不证明RGB可读、动作适用性、稳定抓持或视频必要性。
抽屉task20有6/8episode在stride5采样位置没有双指同时接触；region所属body还可能覆盖整柜、整桌或整架。
因此不把双指proxy及region-body接触直接升格为通用阶段／支撑／前置条件，也不扫阈值或挑task修补。

直接监督生成A的实际响应虽然区别于无关辅助头，仍有明确反例：B=0或rank更新抵消时语义可准确而行为无贡献；
source h已有language时共享A也可能不需要视频。标签覆盖不能解除这两项传递与特异性问题。
撤去“通用接触阶段→语义rank”的直接实施依据；保留分部件变化与左右指物理事实，完整Writer尚无合格启动合同。
新原件约1.36MiB、raw重算及时间对应通过，CLI已退役；本项零梯度、零GPU、零环境步，无held/Test或动作数值读取。


## 83. 真实短段动作效果部分可预测，局部J替代未达登记精度（2026-09-13）

[操作语义§6–7](docs/analyses/operation_semantics_feasibility.md)将旧activation/action effect与真实对象／部件变化分开：
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
H×H对应和冻结回放均不属于该监督。新[设计](docs/designs/visible_object_grounded_writer_design.md)保持完整生成链，
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

[可识别性分析§10–12](docs/analyses/video_information_identifiability.md#10-空间边际监督学到了什么尚不能据它推断什么)
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

[原生纠正传递诊断](docs/analyses/native_corrective_transfer_audit.md)把条件与纠正的联合关系落实到
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

[原生纠正传递§7](docs/analyses/native_corrective_transfer_audit.md#7-完整结果与关闭裁决)完成f39d594f冻结的192套完整rank16 LoRA、
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

[Native Correction Writer](docs/designs/native_correction_writer_design.md)用`ΔW=B(RX)`保留同位置输入与预测纠正的乘积，
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
单独单位A仍162.19。按[设计§9](docs/designs/native_correction_writer_design.md#9-正式学习前的因子单位修正)在正式学习前修正：
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

[Native Correction Writer§10](docs/designs/native_correction_writer_design.md#10-完整有界结果与关闭裁决)完成两臂各fresh200、
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

[固定A误差分解](docs/analyses/native_correction_acquisition_audit.md#6-完整结果与关闭裁决)完成四checkpoint×train24/demo16–19，
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

[获取诊断§7–8](docs/analyses/native_correction_acquisition_audit.md#8-幅度方向分析完整结果与关闭裁决)在3c3c4897登记后，
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
[闭环诊断§7](docs/analyses/native_corrective_closed_loop_audit.md#7-完整结果与关闭裁决2026-09-14)，临时来源准入按关闭合同退役。


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

合同见[冻结行为回放§5](docs/analyses/semantic_path_behavior_replay.md#5-完整结果与关闭)，代码5116deb0、登记3bd9db19。
原件`runs/analysis/semantic_path_writer_20260914/behavior_replay/`保留READOUT、replay_readout、decision、
两份覆盖全部32组的人工观察、固定图像／元数据、128条逐replan轨迹及完整launch／completion。
整体goal未完成，自主授权持续，无active design或在途运行；不从历史关闭段恢复训练。

## 97. 在局部纠正场上限制rank16保留了实际跨episode作用（2026-09-14）

[局部纠正场§5–7](docs/designs/local_correction_field_design.md#7-固定算子完整结果与下一阶段)固定原train24/demo16–19、
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

[局部纠正场§8–10](docs/designs/local_correction_field_design.md#10-完整50100结果与关闭裁决2026-09-14)让同一预测场接受
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
首轮原件为[证据审计](docs/analyses/v52_evidence_audit_20260917.md)和Git `426dc5be:docs/v52_evidence_based_writer_design.md`；
[架构推导](docs/designs/v52_evidence_based_writer_design.md)为同用途持续更新的canonical文档，后续统一设计见§116。首轮审计
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

选择[统一设计](docs/designs/v52_evidence_based_writer_design.md)：native1–9→联合Z/H block×2→双残差写回→native10–18及final norms→
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

## 117. 统一Writer有界窗口结束：早期获取未形成整体保持，1500停止（2026-09-18）

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
| 1500 | 84/400；52/96 | 112/400；56/96 | 5/8；19/24 | 3/18/33/30 | 19/13/15/5 |

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
1200节点更明确的是跨任务迁移与成功保持失败；这些主面板仍没有识别唯一内部失败模块。
900→1200训练段2975.2秒，累计程序训练12235.8秒，4800条件/100800queries；所有checkpoint、496条件及496条评测通过。
按已登记节点再取1500确认回落是否持续，保持训练配置与全部恢复状态；这是一次相邻确认，不以FM下降无限延长窗口。
修正是否开展仍需具体失败证据、主要改变变量、预测与停止标准，最终controls不作为返工依据。

1500训练期间复核任务覆盖这一候选解释：现有allowlist提供71个审计source specs，与train24构成95个名义任务，
但同语言跨场景与spec数量不等于已证明独立的操作教学映射。§68及research_history§6已经记录75任务reader、
73任务共享Writer及meta73/target18完整输出比较；扩展任务不是尚未试过的默认修复。
本轮训练任务表现保持、validation回落只支持优先检查跨任务问题，尚未将覆盖不足识别为首要原因。
当前图不同或训练task较少均不足以单独支持恢复95-task；本次只读复核没有增加梯度来源或改变sampler。

1500完成correct84/400、train52/96，全部496条件fresh生成、496条闭环及30个workers通过。
Validation相对1200 R/G/L43/41/24、churn65、Jaccard .3981，净差95%CI[-2.75,+13.75]pp；
Long1从13→30贡献全部净回升17，其它suite净和为零，breadth仍5/8，Goal3与Long2仍零。
相对A1500少28，差值95%CI[-22.25,+4]pp含零；不能单独宣称该节点统计显著落后。
相对900为109→84，R/G/L56/28/53、churn81、Jaccard .4088，早期整体能力没有恢复。
Train相邻54→52、R/G/L41/11/13、churn24、Jaccard .6308，breadth20→19；
900/1200/1500的Object20/17/13与Spatial15/16/19方向相反，训练侧也有能力交换，不能把全部缺口只归为跨task迁移。
留出训练动作FM1500为.1013617、最近100更新训练均值.1030854，小幅下降未带来持续整体闭环获取。

按统一设计§11预注册1200后相邻节点的停止条款，本次在1500结束，没有执行初始上限内的1800/2100/2400。
依据是900后训练总成功54/54/52停滞、validation109/67/84未恢复且反弹集中单task、覆盖和成功保持仍不足，
不是因为某个loss或内部梯度不好，也不是“完全没有学习”。本次只是该架构/配方/有限窗口的non-pass，
不声称模型完全收敛或全部统一结构没有潜力。没有已识别的工程缺陷或可检验的具体修正，故未启动第二轮重训。
未做无依据的小扫或恢复95-task；相邻正确能力与>145资格未成立，后续same-task-other、learned language/static、
冻结后的最终controls未启动。动态视频必要增量/时序特异性改善未获证明，不将未测写成差额为零；Test保持关闭。

正式1500共6000条件/126000queries，15个完整恢复点保留，训练程序含诊断15073.5秒、allocated峰值34.06GiB。
四节点1984条闭环、120个worker完成记录通过，评测程序4328.5秒。全部训练/生成/评测进程及tmux退出，
干净且已集成的临时frozen worktree已移除，原代码184947cb与formal checkpoint/raw rows继续保留。
最终报告为study的`experiment_report.md`，结构化裁决为`experiment_completion.json`。

原件：`runs/analysis/unified_writer_20260917/unified/paired_readout.json`、`analysis/step{600,900,1200,1500}_*`、
`training/{checkpoints,materialized,evaluation}`与`step600_execution.json`。600评测用五卡×三workers；
避开p2当时升高的其他任务负载，不改变训练的六卡拓扑或配对条件。一次物化设备参数格式错误在CLI解析时退出，
改为`cuda:N`后496条件fresh物化成功，原失败及完成记录均保留。

## 118. A900的过程路径确有行为贡献，语言条件参数不能等同于公共SFT（2026-09-18）

Owner接受约130–140及更可信的视频特异性，允许小幅正常churn，并授权快速原因实验。
固定A900、SFT450与同一aligned raw1000 source，在预注册train24上完成10臂功能和闭环；无新训练、held动作或Test。
新闭环每臂96条，init32–35、teacher46复用四初态；不同于旧A train96视频映射，也不是validation400。
两条固定真实视频为预定Spatial0/Long9的demo46，不按结果选优；在其自身任务上不是错视频。

| 条件 | 成功/96 | Breadth/24 | S/O/G/L（各24） |
| --- | ---: | ---: | --- |
| 正确A | 58 | 21 | 17/19/15/7 |
| 正确A关闭Procedure→AdaLN | 34 | 15 | 11/10/6/7 |
| 固定视频0＋目标语言 | 38 | 19 | 9/12/10/7 |
| 固定视频39＋目标语言 | 38 | 16 | 11/11/8/8 |
| 固定同一LoRA0／39 | 13／10 | 7／5 | 9/0/4/0；1/1/6/2 |
| Source／SFT450 | 12／47 | 7／18 | 2/1/6/3；14/11/15/7 |

关闭Procedure只将同次编译的gamma/beta置零，其余Core、heads和权重不变；正确58→34的R/G/L27/7/31，
净−25pp、task-cluster95%CI[−40.625,−9.375]pp，说明旧路径对当前执行有实际贡献。
它不能证明fresh删除后必然较差，也不把模块名“Procedure”当作已经学会顺序关系的证据。
完整正确/固定0/固定39为58/38/38，关闭Procedure后为34/33/38；正确video的Procedure收益减固定video的收益，
两组配对交互+19.792/+25pp，CI[3.125,38.542]/[6.25,44.792]pp。
因此本面板正确视频的主要增量经旧P/AdaLN表达；旧Core晚期接时序的结构事实仍未被证明是实际容量瓶颈。

固定RGB时保留Writer目标语言，相对固定LoRA分别13→38、10→38，R/G/L11/27/2与6/32/4，两个区间均为正；
执行policy在所有臂都保持正确目标语言，故这一差额定位到参数生成条件化。
575c189a源码中Core evidence含native图文task-span及patch hidden，二者均接受语言条件化；外部text-only只作Q不保证视频不可替代。
这解释“同数据／近参数量的SFT不是Writer上界”，不证明两个函数族的严格包含，也不把两条donor LoRA当最优共享适配器。

功能面板768次固定动作queries×10臂，正确FM.107363→关闭P.116807，24/24任务均变差；
固定video完整为.119416/.122610，关闭P后.116873/.117521，正确关闭P为.116807。
SFT的.101378更低但闭环47低于58，且train96差值区间含零；FM不能代替行为。
queries来自episodes47–49，Writer未训练而SFT见过；这项不对称保留。
SFT与A任务ID/source/38targets一致，但A4×21、SFT24×24、更新次数与LR阶段不同。
A登记参数10,238,778含三Meta和1074个冻结horizon参数，可训练10,237,704；SFT10,297,344。
SFT450条metrics与声明LR吻合，没有发现缩水batch、错误schedule或rank不符；其训练闭环只新增450单点，不能唯一判断过拟合。

既有validation400重新核对correct140/wrong116、SFT400/425/450为85/89/86。
wrong净多31/27/30，task3/31贡献31/29/24；还丢失28/23/28个SFT成功，两个比较区间跨零。
wrong→correct R/G/L92/48/24，净+6pp、CI[.5,12]pp，保留真实视频依赖；不等于时序必要性或相邻稳定。
新固定video在train96为38/38，低于SFT47，故“任意错误视频都胜SFT”不成立；本轮没有直接分解验证116的全部来源。

新统一图多变量改造少31/68分的唯一内部原因仍未定位；不能将本轮冻结消融扩大成所有后继失败的共同根因。
当前支持保留已证有用的功能，并将目标聚焦于正确视频相对语言条件参数参照的可迁移增量；尚未选定新架构或新训练。
240条功能记录、240个shards/960 rows和全部21个worker退出通过；闭环原引擎六卡18workers约1424.64秒。
完整逐task/suite、R/G/L/churn/Jaccard、配对区间、合同及限制见
`runs/analysis/v52_mechanism_audit_20260918/{report.md,functional_summary.json,closed_loop_summary.json,historical_sft_comparison.json,completion.json}`。

## 119. 正确视频增量随Procedure跨Core保留，同视频配套不是主要能力的必要条件（2026-09-18）

Owner同意四格交叉冻结诊断。固定A900／aligned raw1000、target language、teacher46及train24的init32–35，
两条donor仍为task0／39的demo46。encode_task后交换Core memory/token mask与P memory/真实positions/frame mask；
重新计算原Core条件化P读取、AdaLN、postfusion与全部A/B，未混合最终LoRA或改变权重。
新增四臂384条闭环，复用§118三条对角线288行；168条功能记录中三条对角线逐task FM重放差值均为0。
所有任务内query/action位置、state、language、env与policy RNG配对通过；无训练、held动作、Test或模型选择。

| 条件（各96） | donor0 | donor39 |
| --- | ---: | ---: |
| 正确Core＋正确P（CC，共享既有参照） | 58 | 58 |
| 正确Core＋donor P（CW） | 34 | 35 |
| donor Core＋正确P（WC） | 56 | 56 |
| donor Core＋donor P（WW） | 38 | 38 |

固定正确Core，换入正确P为34→58／35→58，R/G/L31/27/3与29/29/6；
差值+25／+23.958pp，task-cluster95%CI[12.5,37.5]／[9.375,38.542]pp。
固定donor Core，换入正确P均38→56，R/G/L32/24/6与29/27/9；
差值+18.75pp，CI[7.292,30.208]／[4.167,34.375]pp。
因此正确P携带的行为增量能在另一条视频的Core下发挥，不要求同视频Core才能恢复大部分能力。
这仍使用目标task自己的正确P，不是跨task迁移实验。

从CC换到WC，两组保留51／48个成功，丢失7／10、新增5／8，净少2；
churn12／18、Jaccard.810／.727。WC→CC差值+2.083pp的CI均跨零，但不据此宣称等价或Core无用。
CC−CW−WC＋WW交互+6.25／+5.208pp，CI[−1.042,13.542]／[−5.208,15.625]，没有明确强配套效应。
WC的S/O/G/L为16/18/14/8与15/19/14/8，breadth21／20；CC为17/19/15/7、breadth21。
CW的S/O/G/L为9/11/8/6与10/13/4/8，breadth均17；完整逐task统计保留在报告。

donor自身task属于恒等输入例外，全部新增恒等条件逐state success与既有CC一致。
剔除自身task后，donor0的CC/CW/WC/WW为55/31/53/35、donor39为58/35/56/38，各92；
两个P条件收益区间仍为正，Core条件收益区间仍跨零，结论不依赖同视频例外。

FM的CC/CW/WC/WW分别.107362545/.119309240/.107362669/.119416467（donor0），
以及.107362545/.122011357/.107556256/.122609594（donor39）。功能方向与闭环一致，损失幅度不能当成功贡献比例。
查询仍为每task32个、episodes47–49、flow seed20260918，与teacher46分离。

该结果进一步支持原P通路及融合已经能传递有益视频差异，当前不支持把晚入口或同视频配套作为优先重构理由。
Core还提供内容及P查询；其视频来源在两组干预中可替换，不证明Core可删或纯语言化。
语言同时影响Core/P，P也可能含多状态／静态语义；没有单独证明真实顺序关系、未见任务泛化或fresh改造收益。
混合memory可能分布外；两个固定donor不代表所有视频，所有新结果只属train24有限面板，不是validation400。
本轮没有识别新统一图的唯一退化原因或指定新架构。
随后Owner纠正：仅据本轮和§118就把后续限制为保留Core/P、只改P或停止整体重构，超出了证据范围，亦不是Owner要求。
这些路线建议撤回；数值事实及局部因果结论保留。后续设计须综合全部历史证据，并满足模块可复制加深和自然扩参的要求。

原runtime来自clean pushed detached575c189a；三个生成worker约194.8秒、allocated峰值9.956GiB，
18个闭环worker完成96jobs／384新rows，首claim至末完成514.90秒，allocated峰值9.338GiB。
全部21workers exit0、无活动进程；干净临时runtime移除，原Git／脚本／96适配器及raw rows保留，新增约477MiB。
原件：`runs/analysis/v52_core_procedure_cross_20260918/{registration.json,report.md,functional_summary.json,closed_loop_summary.json,completion.json}`。

## 120. A匹配learned frame-set：可达142，但未保住相邻能力（2026-09-18）

按专家意见完成唯一fresh集合参照，固定1200停止；不重训A、不并开新图、不按controls选节点。
本轮只删除视频frame RoPE、Procedure causal mask及读取时间寻址；保留真实帧、语言/空间/horizon位置、三Meta、
Core/P中心化与AdaLN、共享完整A/B和A的采样/优化更新语义，使用历史agentview/fixed-mean H口径。
正式运行`a9d86149`，历史A`575c189a`；1200更新、4800条件、100800queries，12个完整checkpoint。
所有400/96行面板完成，source/normalization、环境与policy RNG、teacher映射和全事件匹配检查通过。

| 更新 | A validation /400 | frame-set /400 | A train /96 | frame-set /96 |
| ---: | ---: | ---: | ---: | ---: |
| 300 | 99 | 79 | 36 | 33 |
| 600 | 88 | 109 | 47 | 40 |
| 900 | 140 | 142 | 54 | 49 |
| 1200 | 135 | 118 | 62 | 57 |

主解释900与1200：A→set R/G/L分别102/40/38、90/28/45；churn78、73；Jaccard .567、.552；
breadth6→7、6→5；差值任务簇95%CI为[-4.00,+5.75]、[-12.00,+1.75]百分点（8簇、20,000次、seed20260915）。
两个节点四suite均非零，900的Spatial/Object/Goal/Long为A17/60/30/33、set14/61/38/29；
1200为A14/55/42/24、set8/59/41/10。

set900→1200为142→118、R/G/L91/27/51、churn78、Jaccard .538、breadth7→5；
A为140→135、98/37/42、churn79、Jaccard .554、breadth6→6。总churn相近，差别是set丢失较多、补回较少。
set的Long29→10，主要是双物体放篮28→10（保留8、新增2、丢失20），A同任务33→24（20/4/13）；
Spatial降6、Object降2、Goal升3。train同期49→57（42/15/7）继续改善，不能把validation回落解释成全局未学习。
开抽屉放碗set主节点均0、A0/2；开灶放壶set1/0、A0/0，尚无可靠能力支撑顺序关系判断。

给定播放顺序不是该配方到达约140总分的必要输入，但单点142不证明无序参照相邻等强。
A在1200和Long能力保持有优势，900又不支持A总体占优；结果未满足“有序稳定胜出”或“无序稳定等强”的简单二分。
本轮允许提出旧顺序路径可能帮助特定任务保持的有限假设，未定位唯一内部机制，更不证明删除或加深时序模块会改善。
多帧内容、物理关系及隐含过程推断仍存在；不能推导视频无用、静态图等价或语言独立充分。
单训练seed/8验证任务、区间含零均不构成统计等价；未运行新controls/Test/RL，不升级为最终视频因果资格。
§118–119关于旧P内容被消费的事实继续有效，本轮只补足给定播放顺序训练参照。

[完整报告](docs/review_materials/20260918/frameset_report.md)、[逐任务表](docs/review_materials/20260918/result_tables.md)、
[1984对精简CSV](docs/review_materials/20260918/paired_successes.csv)及汇总/图表已整理；原件保留在
`runs/analysis/a_learned_frameset_20260918/`，旧A原件在`runs/analysis/source_alignment_20260915/A/`。
本轮完成，不自动开展新实验。

## 121. 经最终LoRA的同视频教学：原窗口增量未保持至2100，匹配顺序特异性增量未成立（2026-09-20）

按专家最后修订实现重复完整H50／相邻E读取，保留A的Core/P、条件化融合和完整38-target A/B输出。
每更新四task等权；同一套生成LoRA接受21个跨episode主FM与7个同视频tau1／未来五步教学query，
两项均值按1和1/3共同更新全部Writer及Text/VL/Action Meta，source冻结。没有沿用已撤回的P-only教学梯度。
唯一匹配消融保留图、监督量、噪声流、seed7、学习率与预算，只把辅助query改为同task另一episode。
它识别辅助监督与所看视频的对应关系，不单独识别H/E读取，也不是辅助loss有无的比较。

| 更新 | 同视频 correct /400 | 消融 correct /400 | 同视频 train /96 | 消融 train /96 |
| ---: | ---: | ---: | ---: | ---: |
| 900 | 149 | 125 | 50 | 59 |
| 1200 | 174 | 136 | 67 | 60 |
| 1500 | 165 | 147 | 65 | 62 |
| 1800 | 160 | 136 | 64 | 65 |
| 2100 | 158 | 159 | 67 | 67 |

原窗口三个correct匹配差值为+24/+38/+18，task-cluster95%CI为[0,12.5]/[2,21]/[0.25,9]pp；
1500 other为同视频165、消融155，差值CI[-2.75,7.5]pp。支持该有限窗口的正确能力增量，不能宣称换视频优势稳健。
主组other900/1200/1500为140/159/165，按训练前登记的相邻对规则选定1500后冻结；1200的174不是selected结果。
correct→other1500保留143、获得22、丢失22；同分不等于逐行相同。
历史A对应correct140/135/112，但还同时存在读取、辅助监督和LR尾段差异，只作整体配对参照，未重训旧A或v5.2。

Owner要求观察后续趋势后，两臂从各自完整1500状态继续600步，保留optimizer/scheduler/sampler/rank RNG/world2，
固定LR尾值2.959936384576631e-5，不重启。主组1500→2100 correct165→158，R/G/L132/26/33、churn59、
J=.691、CI[-5.75,3]pp；other165→156，R/G/L136/20/29、CI[-5,-0.25]pp，未见当前尾段改善。
消融correct147→136→159；1500→2100为R/G/L122/37/25、CI[-3.25,10.25]pp，other155→161、CI[-2.25,5.75]pp。
2100消融→同视频correct159→158、R/G/L117/41/42、CI[-5.75,4.5]pp；other161→156、CI[-6.25,3.5]pp。
原1500的正确能力优势没有保持到末点，但159对158不证明消融更好或两组等价；没有新的checkpoint选择。

主组1500→2100 S/O/G/L35/59/48/23→34/61/44/19；消融37/50/42/18→33/68/38/20。
消融净增主要来自Object，抵消Spatial/Goal各少4；不能把总分回升写成全面改善。
两组2100 correct breadth均6/8，分别仅因task1与task32出现一次成功；other breadth仍5。
主组task23/32至2100仍零成功，原1500三个零任务的描述不能不加时间限定地沿用到全部末点。
两组冻结train-action FM1500→2100从.098259576/.098404155降至.097725476/.097496695，
没有同步转为更强未见任务闭环；最早确定的缺口在闭环能力、保持与任务覆盖，唯一内部原因仍未定位。

Owner追加的匹配视频检查固定两组1500，不按control结果选点、修改结构或训练：

| 模型（各400） | correct | other | wrong | shuffled | reversed | 共用source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 同视频 | 165 | 165 | 124 | 113 | 113 | 50 |
| 消融 | 147 | 155 | 122 | 104 | 86 | 50 |

两组correct相对shuffle/reverse均有正的组内差值区间；教学是否增强特异性须直接看matched DID。
以correct参照，wrong/shuffle/reverse的两组优势差分别+16/+9/−9条，CI[-4.5,12]/[-5.5,9]/[-11.25,3.5]pp；
以other参照为+8/+1/−17，CI[-7.5,9.75]/[-6.5,6.25]/[-12.75,1.75]pp。六个区间均跨零。
因此两组都有顺序敏感性，但尚未证明同视频监督增强这些视频／顺序优势。
主组task3 correct/shuffle/reverse35/6/0，Long wrong反而23→33，效应分布不均；不能直接写成普遍过程理解。
共同zero-LoRA面板是冻结source，不是learned language-only/static prior；其DID仅复现原两臂分差。
2100没有新增controls，也不能继承1500的因果资格。

两臂累计4200实际更新、16,800条件访问、470,400主＋辅助query（不重复计续训复制的父历史）；
8400主事件及教学噪声流逐臂一致、source冻结、完整恢复和所有新增梯度组核对通过。
本单相机部分33个正式面板/10,160次新闭环全部完成，共同source只计一次；全池视频、逐行state/RNG、真实RGB重排及workers均通过。
正式主/消融/续训runtime分别39c3919c/bd497edc/d1474ce0，均来自clean pushed detached commit。
只用固定24 train task产生梯度；validation8簇、一个训练seed、20,000次bootstrap且未多重校正限制外推；无Test、RL或融合。

原件位于`runs/analysis/video_teaching_20260919/`；远程保留[原窗口匹配报告](docs/review_materials/20260919/ablation_report.md)、
[完整续训报告](docs/review_materials/20260919/continuation_report.md)、[匹配视频检查](docs/review_materials/20260919/ablation_controls_report.md)、
[逐任务／相邻表](docs/review_materials/20260919/discussion_tables.md)与CSV/JSON。
后续新增相机属于独立fresh对照，不由本段归因；不因有限尾段结果宣称所有更长训练无效，也不自动恢复本尾段。

## 122. 同视频教学的双相机对照未改善当前配方；整轮证据已完成（2026-09-20）

Owner凌晨授权在资源与时间允许时补双相机。经最长105帧及四卡真实吞吐／完整恢复检查后，
执行唯一fresh1500：仅给Writer添加同步eye-in-hand教学RGB，执行policy本来就使用双相机。
source、seed7、train24、参数量、6000事件、实际query及噪声、21＋7监督、AdamW与LR预算保持匹配；
物理world2/frame16改为world4/frame8，每rank policy microbatch16，四任务等权更新语义保持。

| 更新 | agentview correct /400 | dual correct /400 | agentview train /96 | dual train /96 |
| ---: | ---: | ---: | ---: | ---: |
| 900 | 149 | 117 | 50 | 52 |
| 1200 | 174 | 108 | 67 | 64 |
| 1500 | 165 | 108 | 65 | 67 |

三个correct净差为−32/−66/−57，task-cluster95%CI为[-14.5,-1.5]/[-31.5,-4.25]/[-27,-4]pp。
1500单→双保留93、获得15、丢失72，churn87，J=.517；S/O/G/L35/59/48/23→10/43/36/19。
双相机末点correct breadth6/8仅因task1出现一次成功；不能以覆盖计数掩盖既有能力下降。
fixed1500 other165→105，保留88、获得17、丢失77，CI[-27.75,-2.75]pp；dual breadth7/8中task1与23各只有一次成功，task32仍零。

dual900→1200保留77、获得31、丢失40；1200→1500虽同为108，仍保留78、获得30、丢失30，churn60、J=.565。
dual correct→other108→105，保留76、获得29、丢失32，CI[-3.25,1.75]pp；总分相近不等价于条件行为稳定。
train成绩52→64→67，固定train-action FM900/1200/1500为.104100628/.099837918/.098245136；
末点与agentview .098259576接近，训练任务提升及FM下降没有转化为未见任务能力。
没有唯一定位表示、泛化或优化方面的内部原因；真实相机／完整帧、事件、source冻结、四组有限正梯度及全部worker核对均通过。

当前配方的双相机补充不构成有效改进，不宣称所有双相机方法无效，也不归因于某一接触、遮挡或数值机制。
只有一个训练seed、八个validation task簇，bootstrap20,000次、seed20260915且未多重校正。
没有双相机wrong/shuffle/reverse、续训或新选点；不继承单相机1500的controls资格。原selected1500保持冻结，无Test/RL或自动新实验。

双相机1500更新计算22233.25秒、均值14.822秒、allocated峰值28.175GiB；15个完整checkpoint保留，formal runtime为clean pushed detached35124aa9。
新增7面板/1888行全部审计通过；整个study在08:50完成40面板/12,048条新闭环、5700实际更新、638,400queries与57个唯一arm/step完整checkpoint。
续训复制的父历史不重复计数，共同source只计一次；08:51两节点核验本轮无运行中的GPU或训练／物化／评测作业。

完整[双相机报告](docs/review_materials/20260919/dual_camera_report.md)、[总报告](docs/review_materials/20260919/final_report.md)、
[专家提示词](docs/review_materials/20260919/expert_discussion_prompt.md)、[完成清单](docs/review_materials/20260919/completion.json)
与逐task/suite CSV、训练趋势、相机示例及配对原件共同保存。本轮结束后的研究选择交Owner与专家讨论，不自动启动后继训练。

## 123. 冻结1500的Test优势未保持：82对74，未达+40门槛（2026-09-20）

冻结Source1000、MT-BC425与EMBER1500的Test8分别78/74/82，各400；EMBER相对MT-BC仅+8（2pp）。
任务簇bootstrap95%差值区间[-8.51,11.25]pp，配对保留/获得/丢失50/32/24，churn56；breadth为Source5、MT-BC5、EMBER4。
Spatial为6/5/22，Object0/15/0，Goal36/43/48，Long36/11/12（各suite100）。
局部Spatial增益不足以支持广泛迁移优势；Object的15条损失与Long相对Source的24条损失必须同时报告。
1200行完整、18个worker正常退出、source/normalization与state/RNG配对、50teacher无放回调度已核对。
这是当前冻结方法在固定Test面板上的有效负结果，不自动指向工程缺陷或否定全部视频到LoRA方法。
没有Test视频controls，不能断言Test特异性成立或消失。没有根据结果更换checkpoint、增seed或调架构。
Owner选择先看[图文报告](docs/review_materials/20260920/test_capacity/report.md)并与专家讨论，所有下游暂停。

## 124. 覆盖重训检验的是训练支持内组合迁移（2026-09-20）

Owner采用新的单次24/8/8＋12 source辅助划分，两方法统一36任务等权；Source71与normalization复用。
[覆盖审计](configs/libero_24_8_8_coverage_v1/coverage.md)核对BDDL语义、对象/装置及训练HDF5 metadata；不读取held数值标签。
Long9 cup→microwave仍是新组合/几何，训练支持来自L4操作cup及aux33关闭microwave、aux3放入后关闭；
不能声称训练包含向microwave放入物体。Goal6原始规格是On cream_cheese bowl，不是In。
Long8初始炉灶已开启，Turnon证据实际来自Long2；Spatial9柜顶取回与aux15放置方向不同。
这些区别保留为泛化边界，不能用task名或对象词匹配冒充逐任务完整技能已训练。
新方案同时改变split、辅助任务支持、曝光时钟与动态停止，不支持把未来差值唯一归因于其中一项。
旧Test78/74/82与失败门槛保留；新Test并非项目历史从未接触的数据。当前profile只证明吞吐、梯度和恢复，不证明新模型性能。

## 125. 修订四臂短窗已完成；终点事实不足以由执行agent选择分支（2026-09-21）

专家修订取消原E2，复用完整E0/E1，只从O1200/N1800完整Writer与AdamW状态分别以高/低固定LR运行
当前36任务事件1801..1836。四臂各36更新、144实际task events、36条terminal compact probe及四个held任务
×state0..3闭环全部exit0；合计144 training steps、576 task events、144 probes、64 rollouts。四个九更新周期均
覆盖36任务一次。闭环只在每臂Spatial3-state0和Long1-state0保存图像，因此64条轨迹中8条full、56条compact；
既有E1另只读导出16条轻量双相机视频和5536个实际执行归一化动作，不做新forward或rollout。

相同16条件的小闭环成功数为：O1200父节点11，O-H 11，O-L 10；N1800父节点8，N-H 7，N-L 10。
逐任务顺序global3/11/26/31为：O1200 2/3/4/2，O-H 3/3/4/1，O-L 3/3/4/0；N1800 0/4/4/0，
N-H 0/3/4/0，N-L 2/4/4/0。terminal actual10-step flow first5 MSE均值为O-H .054112、O-L .053502、
N-H .052721、N-L .050463；父O1200/N1800为.054557/.050687。相对父动作delta MSE均值依次为
.007083/.003248/.004216/.001584。

这些是四任务×四状态和36个固定动作query的诊断事实，不是Validation400、模型资格或因果归因。
高/低LR在两个父状态上的闭环方向不一致，且probe MSE与闭环成功不能互相替代；执行agent不据此选择分支、
续训或启动Test/FT/RL。脱敏原始表、completion、轨迹索引和轻量视频保存在
`docs/review_materials/20260921/writer_stability_diagnostics/corrected`，后续解释与决策交Owner/专家。

## 126. 新B空间改变旧更新但未造成净成功下降；N1000系数更趋同（2026-09-21）

O1200/N1000/N1800的q-B和v-B末层BF16有效矩阵均为完整数值列秩216。O1200与N1000的主角余弦平方
均值在q-B为.1900、v-B为.8496，说明q-B列空间差异明显更大。冻结O1200 LoRA投进N1000空间后，q-B
有效更新平均范数保留.6948、`rho=.7188`，v-B分别.9632/.2674；实际方向已经显著改变。

固定held global3/11/26/31各state0..3的O1200/SELF/NEWSPACE/SHRINK为11/11/11/9。NEWSPACE相对
O1200为R/G/L 8/3/3，SHRINK为8/1/3；两者逐层有效更新范数匹配，NEWSPACE没有出现净损害。因此本轮削弱
“N1000 B空间排除旧有效方向”的解释，也不能把SHRINK下降唯一归因于幅度或方向。16条件面板存在SELF一得一失，
不把两条差异外推为总体方向优势。

四个Train任务×demo46/48的A1显示N1000 rank-centered code能量约.0080–.0095，O1200约.0535–.0539；
N1000 rank平均余弦约.998，O1200约.984，B与BA第一奇异方向能量也更集中。这个描述支持把系数映射趋同列为
候选定位，但不证明其造成完整Validation不稳定，也不授权split-head、白化、正交或对比损失。初版A1的词典序
layer误配已封存；修复后重建误差最大.001745，A0/A2不受影响。完整边界见对应报告。

## 127. 夜间有界探索未验证辅助目标修复；视频作用须联系实际行为（2026-09-22）

旧MT-BC按每task4800 queries预定补测step200=102/400，高于旧晚期425的89；新非峰成熟节点仍119–138。
共同五held任务的旧200/旧EMBER1500/新M300/C600为91/148/134/130，MT-BC净增43中Spatial3占38；
EMBER减少18主要来自Object1−9、Goal6−14。该分解保留训练池/时钟差异，不是单独划分消融。

冻结三模型实际LoRA与flow检查未支持A/B整体相消、视频只改变Proxy而不改变BA、或仅tau1拟合提高。
C600/C1200换视频后BA相对变化均值.2362/.1827；动作差异均值缩小21.8倍，中位数仅2.33倍，少数夹爪分叉
主导均值。C1200六个tau演示误差均改善，不能用内部敏感性或平均MSE恢复代替正确视频的闭环收益。
M300最优有效rank16保留99.68%BA能量，训练侧16闭环8→8，R/G/L=6/2/2；不支持单纯rank差为主要瓶颈。
因子C600有一条旧D1参考动作maxabs=.1148超过预定.05，明确保留`complete_reference_mismatch`；
本次独立flow复核用于同次配对分析，不替换历史原件或追逐逐bit一致。

唯一54更新P/J/F诊断中，F仅将辅助7从tau1/前5改成随机tau/全50，保持episode/noise/权重/Adam与事件601–654。
全36训练任务×2状态的P/J/F/M300为39/46/44/44；目标24为18/23/23/21（各48），辅助12为21/23/21/23（各24）。
F−J为−2/72，R/G/L=36/8/10，task-cluster95%CI[−15.28,+9.72]pp；目标24为0，CI[−18.75,+18.75]pp。
原目标J的Long+5同时有Spatial−4；未满足F预登记明显信号，不改选J或据此批准fresh。短程续训阴性不否定fresh可能性，
也不证明架构性能上限。主项已对21个不同episode同时监督同一视频LoRA，不能再把“缺少跨多episode主监督”当成事实。

Spatial3的wrong donor旧13/新11混杂经唯一冻结2×2补齐：O1500两donor均0/50，C600分别40/45，correct41。
特定donor变化不足以解释旧新差异；该sealed post-hoc检查仅校正历史比较，不进入模型/架构选择。
同任务视频信息是否改善性能仍是核心目标；不以放大敏感性、损害wrong或先建公共base替代Owner的下界与视频利用论证。

追加唯一任务梯度隔离反事实：从同一C600完整状态，每个固定Train8任务单独保留events601–654中自己的新梯度，
其余梯度置零但保留全部54个AdamW step；Z分支全程零新梯度。同一16条件P/J/F/M300为3/7/7/8，各task独立副本
共成功4条，Z为1条。J→独立副本R/G/L=3/1/4，唯一新增Spatial5也在Z中出现；J的Goal5、Long4/7收益消失。
不支持去掉其它任务的新梯度即可修复当前模型。独立副本不是一套部署方法，不能把4/16作为候选方法成绩。
零梯度仍有历史动量与decay，Z/J参数位移L2为2.249/4.546；Goal0/Goal5副本各一次裁剪而J无裁剪。
该干预同时改变优化轨迹与步幅，且每task仅2状态，不能据此排除所有任务干扰，也不能宣称Adam错误或支持直接更换优化器。

本轮852条新闭环与J/F的108次完整诊断更新已经完成；隔离检查另有48次真实task梯度事件、486个含零梯度optimizer step。
正式训练、Test、FT、RL未启动；没有已验证修复或唯一根因。
原始行与图保留于`/data0/user/ymdai/ember_runs/overnight_root_cause_20260922`；按Owner最新要求直接对话汇报，后继正式训练须确认。

## 128. 一小时追加检验：静态末帧信号可定位，但不能直接解释全部能力缺口（2026-09-23）

Owner追加23:03–00:03一小时分析/小试，正式训练仍须先确认。八个新旧共同Train任务global5/7/12/14/20/25/34/37，
固定state32–35、teacher46、wrong donor0/demo46及目标语言/RNG7；同任务另一正确视频为demo48，非正式400或新任务泛化测试。
旧A900同条件既有CC/CW/WC/WW/OFF为17/8/16/10/11，新测O1200为20/12/19/13/20，C600为8/8/7/8/6（各32）。
O1200→C600正确条件R/G/L=8/0/12，OFF为5/1/15。保留正确P、替换Core在旧O为19而C为7；
当前正确P没有重现旧模型相对wrong的正收益，且关闭P后仍有大幅能力缺口。不能把全部下降归因于部署时P调制的单一损害；
这也不能排除训练期间P梯度通过共享参数影响其它通路。O1200与C600使用同一架构，训练池、曝光、监督配方不同，不能据此唯一归因。

固定同一首帧真实native E/H到全部原位置后，Procedure前F−1项完全相同，末项因缺少after E而不同。
若差异为delta，中心化后前项为−delta/F、末项为(1−1/F)delta，故静态输入也产生秩1时间信号。
仅屏蔽末个P可使该静态P/slots/gamma/beta归零，同token数的屏蔽中间P不能；最后真实画面仍在Core和倒数第二步after E中。
这一现象旧O也存在，不能称当前退化的特有bug。FP32均值干预的最终CC/WW LoRA与原C600相同，未构成功能干预，剩余冗余评测停止。
冻结屏蔽末P的correct/wrong为9/7，屏蔽中间P为5/7；前者相对原correct8仅R/G/L=6/3/2、净+1，任务簇区间跨零。

两模型各32个完整Writer主梯度的2×2分解中，video/query平方分量比C600=.02274、O1200=.00646；
这只是所测八任务的梯度波动诊断，不证明全局无视频信息或无局部梯度冲突。当前主21查询确实来自21条不同episode，
而共享动作query下平均两条视频FM的期望目标仍等于单视频FM；恒等式中的输出差异项不能被当成新增期望一致性正则。
结合CV-CSD/DJNFR已有接近负例，没有根据该项启动多teacher训练。

唯一新训练为C600屏蔽末P后的L18：原601–618事件、18次四任务更新、1512主+504辅查询、原固定LR和Adam，所有Writer/Meta共同更新。
L18完整correct/other/wrong=15/11/8，父C600为8/未测/8；正确相对父R/G/L=6/9/2，wrong净0。
L18 correct相对wrong为R/G/L=8/7/0，但必须与原目标同样18步的J18比较，不能把继续训练的收益自动归给边界修改。
J18完整correct/other/wrong=15/10/9。J18→L18正确R/G/L=14/1/1、净0，另一正确视频7/4/3、净+1，
wrong6/2/3、净−1；三项任务簇区间均跨零。因此L18相对父的+7不能归给边界修改，未验证到改善两个目标的新方案。
原目标J18也恢复训练侧视频净收益，但这不是新任务泛化证据，不能据此批准正式续训或声称Validation超过MT-BC。
该小时完成640条新的登记比较条件、64个完整Writer梯度与18次新更新；复用A900的160行，共800行进入配对汇总。
另有60条功能等同FP32条件及5条分卡衔接时已完成的冗余条件保留，未进入方法比较；不以完成量代替根因结论。
O1200的OFF原进程40分钟退出124，保留28条完成轨迹并仅补余下4条；C600路径与L18通过同bank分卡完成，原退出143保留。
逐臂32条件、语言、环境种子与policy-noise共同前缀核对通过。00:01确认两节点本小时作业全部退出，没有正式训练、Test或RL。
内部脚本、原始行、分卡/恢复登记与汇总保留在既有study的`one_hour`；不新增单独报告文件。


## 129. 当前C600同起点局部学习不复现“自由A/B明显优于完整Writer”（2026-09-23）

Owner本夜授权的新Train-only矩阵，在共同8 tasks [5,7,12,14,20,25,34,37]、teacher16、states32–35上，
父C600／private完整Writer／free实际compiler code／固定加权RMS壳面code／free完整A/B分别为9/16/11/11/14（各32）。
所有学习臂从相同C600权重与实际生成LoRA起步，fresh AdamW、相同24次task-local主21＋跨episode辅7查询/noise；
直接优化与code仅作oracle，不作为部署方法。完整Writer对父R/G/L=6/10/3，自由A/B为5/9/4。
独立demo46–49的前5步10-flow动作MSE为.10837/.10041/.11392/.11394/.11357；不以此替代闭环。

四个学习臂首步在同两个训练query上做预注册有界动作RMS校准，不按成败选LR。direct实际/Writer步幅比.811–1.115；
code及code_norm在task12仍约.442、task20约.692/.646，其余约.801–.927，因此不能宣称完全等步幅或模型类上界已证明。
该实际预算未出现“直接A/B明显胜过完整Writer”；freecode与保径向约束code净分相同。既有9月15日freeq18/freeAB46的结论
不能直接套到当前模型，当前证据不支持据此扩宽head、删除slot norm或宣称视频前端无法获取能力。剩余共享训练与跨task泛化
问题未被此局部oracle解决，也不因privateWriter涨分就称视频有效增量已建立。
八个worker与五臂160条闭环完整exit0，task/state/language/env/policy噪声共同前缀核验通过；有限teacher复用范围已登记。
原件在`/data0/user/ymdai/ember_runs/overnight_root_cause_20260922/capacity_route_20260923/aggregate.json`及其registration、worker原始行。

同一C600起点的额外更新拆分完成8个真实task梯度、原Adam及fresh Adam各七个冻结组合，共224实际动作查询。
原Adam中new-head/old-code的动作变化在7/8任务大于old-head/new-code，但原头更新也将同task查询动作MSE均值从.10959降到.09479，
完整更新.09304，而仅将W2位移乘1/sqrt216的完整更新.10231；这只支持头部相对功能步幅较大，尚不支持它有害或应删除。
该梯度是单task单位权重，不是生产四task平均，已明确保存此限制。后继H54仅测试这项步幅平衡能否改善真实闭环，不据几何直接开fresh。
原件为同study的`head_step_20260923/results`；H54实际状态看progress，不由此段自动恢复实验。

H54后续完整闭环否定了本次具体修正：原C600 Adam、原601–654事件及54更新不变，仅八W2实际位移乘1/sqrt216。
Train36×2为46/72，与J54原更新同分，配对40保留、6得、6失；相对父39有净+7，但不能归给W2缩放。
共同Train8×states32–35的correct/other/wrong为11/9/13，J54为15/12/15（各32）；正确对J54保留8、得3、失7。
所有七H54与两J54 worker均exit0，teacher、task/state/language与env/policy RNG前缀核验通过。
因此不把较大的head功能步幅当作已证实有害因素，不以本次缩放配方启动fresh；不外推为所有输出参数化都无用。
原件`head_step_pilot_20260923/aggregate_panels.json`，raw rows保留旧meta标签差异说明，不用标签文字代替当前36任务allowlist。

## 130. 同更新的任务覆盖和总查询量是两个不同的采样干预（2026-09-23）

MT-BC300所属训练每步36tasks×16queries，Writer为4tasks×(21main+7aux)，任务均值权分别1与1/3。
两者同时改变参数化、每步任务覆盖、总标签量及优化成本，不能仅凭基线强就归因任务冲突。
9月9日8e45a27b曾登记4×64→24×10/11的同256query对照，但随后停下、未执行；旧v5.2/v6 task-complete还改变Adam步数/
标签量/LR阶段，结果方向不同。旧强模型同样使用4task，故少任务并非充分失败原因，但当前变量尚未有匹配反证。

当前C600八任务固定teacher16、两套独立21主+7辅查询的八末层W2梯度，query噪声迹估计.09217，扣除该噪声后的task间迹.03743。
若以任务条件独立、query噪声随数量反比的简化估计，固定总标签4→12条件使方差从.03240降到.02616，约19%；
同task两组梯度cos为.106–.699。这不是完整Writer/Adam的统计，辅助位置共享episode，有限八task也不能代表全部36任务。
其作用是避免把所有波动叫task冲突，且区分两种可检验干预：D54把同84+28标签分到12条件；B54保持这12条件并恢复各21+7，
得到252+84标签。两者共同保持原C600 Adam、54步、原事件601–762、每条件均值权重，B相对D只增加查询监督。
训练侧闭环尚待独立裁决，不因方差、loss或计划已实现而批准正式训练。原件`task_diversity_pilot_20260923/variance`。

D54全部七个终点worker完成exit0：Train72为53，原J54为46、父C600为39；对J保留42、增加11、丢失4，
任务覆盖31/36对28/36。suite Spatial/Object/Goal/Long/额外meta为8/8/7/7/23，J为5/5/8/5/23；
增益不来自已经很高的额外meta任务。共同Train8视频correct/other/wrong为17/16/14，J为15/12/15（各32）；
正确对wrong保留13、得4、失1，另一正确视频相对J保留10、得6、失2。正确两臂一起提高而wrong未提高，
是当前固定标签预算调整的初步闭环正证据，但视频净差仍小，不能宣布共享训练根因已证实或held泛化已解决。
在读取此结果后登记一次独立复核：相同八任务、states36–39、correct47/other49/wrong donor0 demo47，
固定J54/D54各三臂，共192行；不替换原面板，不改checkpoint，不重训，不读部分分数。B54仍按原168行终点裁决。

独立复核完整后，D/J的correct为13/11、other15/12、wrong11/10；合计两面板D30/31/25、J26/24/25（各64）。
正确对J累计18保留/12得/8失；other为19/12/5；wrong19/6/6。D的正确对wrong为21/9/4，J为18/8/7。
新增面板单独保留，不能把事后扩展说成最初固定的64条件确认性试验；有限Train证据只支持具体干预可正式检验。
B54完整Train72=47（对J39保留/8得/7失），video32=19/17/14；三倍标签未保留D在全部训练任务上的53/72，
虽局部视频净收益较强，整体绝对能力优先，唯一fresh选择D。两种干预不支持“梯度噪声越低必然越好”的单调解释。

## 131. 固定标签的十二条件fresh未复现训练侧小试的泛化收益（2026-09-23）

在§130正向小试及独立条件复核完成后，唯一fresh于02:51启动，训练runtime为clean pushed detached0684ee05。
完整Writer和三组Meta、Adam/scheduler/sampler均从头开始；Source、架构及主/辅助期望目标不变。
1200实际更新、14400个task/video条件、100800主＋33600辅助queries，平均14.602秒/步；训练exit0，12个完整checkpoint保留。
新协议correct Validation400的200/400/600/800/1000/1200依次103/117/142/141/138/107，各节点完整400唯一条件、12 workers均exit0。
按先验规则选唯一600步142，低于既定MTBC300155和旧跨episode辅助C600154，未达180实用门槛；不补中间点或追加训练。
所选点Long1/Long9/Goal3/Goal6/Object1/Object6/Spatial3/Spatial6为26/0/0/44/30/5/11/26，各50；对MTBC保留87、得55、失68。
1200点只有4任务有成功，相对600保留82、得25、失60；这里报告交换事实，不另加Owner已取消的相邻稳定性接受门槛。

一次独立冻结fresh400的Train72检查使用与C600/J54/D54/M300相同36任务、state0/1、teacher46/48：
fresh400为47/72、breadth30；C60039、J54=46、D54=53、M30044。fresh400对C600保留36、得11、失3，对M300保留37、得10、失7。
该同一fresh checkpoint的正式Validation只有117，支持区分已见任务获取与未见任务迁移；小面板与不同训练历史不构成唯一根因证明。
小试D54从旧C600完整Adam起点继续54更新，而正式模型fresh并有150更新warmup；本次没有证明续训收益可以转移到从头学习。
因此任务覆盖改法在训练侧有局部实证，但没有成为本轮fresh泛化修复。梯度方差式不能替代实际闭环，更不能由此宣布架构性能已到上限。

训练与模型选择已停止，600步的method freeze和原始400行已封存。最终correct/other/wrong已完整完成，见§133；shuffle/reverse在完整面板前由Owner停止，
不得将D54的30/31/25或旧模型的视频结论移植到这个fresh模型。本轮尚无Test/FT/RL，也未把诊断权重提升为另一正式候选。
原件：`/data0/user/ymdai/ember_runs/coverage_task_mixing_20260923`；小型PPT数据与图位于`docs/review_materials/20260923/overnight_results`。

## 132. Frozen MT-BC300 在历史暴露 Test8 上高于 Source，但增益小于 Validation（2026-09-23）

Owner授权的 coverage 划分 Test8 baseline 已按相同400个task/state、环境与policy RNG及噪声共同前缀完成配对。Source1000为75/400，MT-BC300为121/400（+46，+11.5个百分点）；breadth为4/8→7/8，success-set保留/新增/丢失46/75/29，churn104/400（26.0%），Jaccard .3067。按8个任务聚类的增益95%描述bootstrap区间为[−0.5,+22.75]个百分点，包含零且范围较宽，不构成等效性或显著性结论。

同协议Validation Source/MT-BC为51/155，增益104（+26.0个百分点），R/G/L=36/119/15，breadth4→6。Test的MT-BC增益比Validation低14.5个百分点；对不重叠任务集独立重采样所得95%描述区间为[−41,+9.5]个百分点，同样不能据此断定泛化差距。本Test任务此前已有历史暴露，因此它是Owner授权的提前baseline测量，不应描述为新盲测；Test结果未用于checkpoint或方法选择。按suite的Source→MT-BC为Test Spatial3→20、Object0→31、Goal36→46、Long36→24。全量逐task/suite与配对行见`/data0/user/ymdai/ember_runs/coverage_baseline_test_20260923/analysis/`下CSV/JSON。

## 133. 最新零交互结果、baseline Test与四臂定位边界（2026-09-23）

夜间固定标签12条件fresh的唯一选定600：Validation correct/other/wrong = 142/132/105（每臂400）。
correct−wrong净+37，保留91、correct-only51、wrong-only14；correct−other净+10，保留113、各独有29/19。
两条正确视频均高于wrong，但三臂绝对分数均低于旧C600的154/160/153，不能宣称绝对能力与有益视频利用已共同修复。
本轮Test、FT、RL未执行；shuffle/reverse由owner暂停，无完整成绩。

按独立授权补测的Source1000/MT-BC300：Validation51/155，Test75/121（各400）；Test内R/G/L=46/75/29。
Source在Test更高而MT-BC更低，不支持“Test统一更难”的概括，也不能假定MT-BC没有迁移/保持问题。
两个任务集合不同，8任务簇和总分差不足以判定划分有效/无效；Test不用于重选点或修订本批任务/配方。
完整原件：`/data0/user/ymdai/ember_runs/coverage_baseline_test_20260923`；交付commit`7b18030c`。

已登记但暂停的conditional_compilation四臂采用fit28/diagnostic-held8，官方24/8/8不变；旧MT-BC训练过这8个任务，
故A是相同新诊断划分/查询/优化时钟下的直接LoRA参照，不是重新发现MT-BC方法或以rank解释缺口。
A/B改变参数化和条件映射，B/C还改变活动模块，C/D改变tau＋horizon目标组合；固定数据不能单独检验数据构成原因。
该矩阵只是定位学习环节，没有产生任何新训练结果，更不能宣布唯一根因。未验证实现封存在`73267f53`，最新状态看progress。

讨论中的生成Jacobian、纯噪声端点动作回归及任务关系覆盖解释记录于concept；它们是可检验机制框架，不是已证实结论。
不得将“需要以后用有判别力的干预补齐”写成已经获批的第5臂或自动后继训练。

## 134. 接管只读复算：视频差距扩大伴随正确能力下降，不是已验证修复（2026-09-23）

新主讨论从canonical results.json复算原cross-episode辅助C与夜间12条件fresh N各六correct节点及selected600的other/wrong，
共16个已完成400面板；全部400唯一task/state、每task50视频各一次、worker exit0。跨运行selected600的三组1200对条件
另核对语言、环境/policy RNG共同前缀、condition ID、teacher真实帧元数据、Source及冻结normalization内容一致。

| 同一评测臂，C600→N600 | 成功数 | 保留/新增/丢失 | churn |
| --- | --- | --- | --- |
| correct | 154→142 | 94/48/60 | 108 |
| same-task-other | 160→132 | 88/44/72 | 116 |
| cross-suite-wrong | 153→105 | 81/24/72 | 96 |

correct−wrong从1增至37的变化可分为correct少12、wrong少48；另一正确视频也少28。它保留N模型内部正确视频有优势的事实，
但不能当作相对C同时改善绝对能力与有益视频利用的证据。总分接近也不代表同一成功集合；correct仍有48得60失。
这是跨运行描述，训练条件曝光/每条件查询和初始化学习历史不同；controls没有用来修改当前四臂科学参数或选点。

同时重读7月/8月四个原始匹配访问面板，复核v5.2 old900/TC150=132/51、v6 old900/TC150=95/111，
保留Adam次数、LR阶段及查询预算混杂。C600局部接口五臂160条原始行复算为parent/private Writer/free code/code_norm/freeAB
=9/16/11/11/14（每臂32）；有限task-local预算及动作步幅校准不构成整体容量上界，也不支持由旧freeAB正例直接判当前head受限。

本轮无新增forward、梯度、rollout或Test读取。复算源路径与逐面板计数保存在
`/data0/user/ymdai/ember_runs/conditional_compilation_diagnostics_20260923/coordination/scientific_recheck_20260923.json`。
机制预测只进入task_plan；这些机械复算没有确定统一根因，也没有新增第五臂授权。

## 135. 四臂完成：语言生成学到能力，视频条件与保持仍未共同改善（2026-09-24）

Study：`/data0/user/ymdai/ember_runs/conditional_compilation_diagnostics_20260923`；运行commit`43d801b1`。
主讨论从54份原始results.json独立重算13200行，核对唯一task/state、worker exit0、clean运行commit、
共同语言/env/policy RNG前缀、跨节点teacher映射及C/D完整重编码的other/wrong条件；全部与机械汇总一致。
四臂均1260更新、141120动作查询、12 checkpoints；选点严格按六个held400 correct节点最大值，同分取早。
复算脚本与结果在study的`coordination/registered_batch_recheck_20260924.{py,json}`，未新增模型forward、rollout或held动作读取。

| 固定模型 | selected | held400 | seen64（同点） | Source相对保留/获得/丢失 |
| --- | --- | --- | --- | --- |
| Source1000 | 固定1000 | 58 | 9 | — |
| A直接rank16 | 420 | 108 | 19 | 49/59/9 |
| B语言Writer | 630 | 124 | 27 | 48/76/10 |
| C视频纯FM | 420 | 110 | 22 | 34/76/24 |
| D视频辅助 | 1050 | 123 | 31 | 40/83/18 |

同节点held差值B−A为`[4,-10,20,13,16,17]`，C−B为`[-2,12,-19,-13,-8,-19]`，
D−C为`[25,-3,3,7,13,-9]`（210..1260）。B后四节点超过A，且seen学习没有普遍弱于A，
削弱“生成参数链普遍无法获取直接LoRA能力”的强解释；不能据此宣布优化/decoder无问题。
C@420与B@630都获得76个Source失败条件，但不代表同一成功集合：B→C保留82、获得28、丢失42。
C并非完全不会学习，而是获取与保持的分布不同。所有最佳点不同训练时长，不能只由最佳排名判断因果。

C selected correct/other/wrong＝110/98/126；D＝123/110/80。C→D的三种输入分别变化+13/+12/−46，
correct−wrong差从−16变成+43，其中46/59来自wrong下降；两种正确输入也有小幅正变化，但没有建立超过语言B的收益。
B selected→D selected保留95、获得28、丢失29；总分−1不等于策略等价。
D内部correct比wrong多43（共同63、correct-only60、wrong-only17），other比wrong多30，
说明这批冻结条件中正确视频确有相对错误视频的优势；尚不说明动态内容必要，也不构成绝对能力与有益视频增量的共同修复。
按8任务簇的描述性95%区间，D correct−wrong为[2.25,21.75]pp、other−wrong为[1.75,14]pp；
D−B selected为[−5.5,4.5]pp。单训练seed、少任务簇、选点后的条件性区间均限制外推；区间未校正六点选最大。
correct→other下降12/13也可能包含选点对固定state-video配对的乐观偏差，不能单独宣称视频质量导致不稳。

D1050→1260，held123→91，保留63/得28/失60；seen31→36，保留26/得10/失5。
held任务净变化：Goal21−25、Spatial0−10、Long36/38合计−8、Object14−2、Object15+13。
这不是全能力统一崩溃；损益集中且同时存在，不能由下降直接断言梯度冲突或唯一遗忘机制。

登记的actual10-flow probe每模型held32/seen64条。held前5步真实动作MSE为
Source .05498、A .04407、B .04357、C .03974、D .04213；seen为.07576/.05799/.04395/.04450/.03558。
C的held平均误差更小却没有更强闭环；D的seen误差更小也不能推出held更稳定。
这些是固定演示观测、少量位置的同动作目标误差，未识别实际状态分布、对象离散选择或后续恢复原因。

固定full cases中Goal21/state0：B630成功，C630可见将盘子移到炉子上，D630成功，D1260失败且碗仍在原处。
另一固定state25四者均成功，D630直到295步才成功；因此同任务也不是统一行为模式。
图像只提供对象选择与状态历史的竞争线索，不能直接等同grounding根因。9月14日Semantic-Path全128回放已呈异质失败，
9月13日空间Q/K监督改善一般能力却未恢复Goal；这两个历史边界继续有效。
本次图像索引：study下`coordination/goal21_fixed_state{0,25}_timeline.png`，源为原始full轨迹，无重新rollout。

裁决：关闭四臂定位批次；不把D视作已验证修复，不立即增加一条训练配方。下一项登记为冻结前段轨迹与视频条件干预，
在已暴露的两个Train诊断任务上，区分共同状态下的条件作用和早期动作造成的历史效应；设计见`docs/designs/frozen_prefix_causality_design.md`。
这些机制仍待干预证实，当前没有统一根因。

## 136. 冻结前段干预完成：早期机器人接近状态有因果作用，尚未识别生成偏置的根因（2026-09-24）

Study：`/data0/user/ymdai/ember_runs/frozen_prefix_causality_20260924`；唯一正式实现`1f583d9e`。
20面板1600唯一分支、400旧k0引用、64固定双相机full cases齐全。资源准入曾中断，恢复后未重跑已完成分支，
所有正式面板worker exit0；完成信号`analysis/completion.json`。无训练、新held expert action或官方Validation/Test读取。
主讨论独立从20份原始results核对1600分支、原bank/teacher/RNG、同锚历史配对，并重算40组登记R/G/L及20000次paired bootstrap；
与执行者汇总一致。重放保存state与同锚四followers的sim/controller/动作/对象/EEF/夹爪历史差异均为0。
复核原件`coordination/completed_recheck.{py,json}`；全部64固定case拼图已查看，见`coordination/*_fixed_overview.png`。

下表每格为50初态成功数；是冻结的混合执行分支，不是EMBER部署分数。

| 任务 | 前段 | 截断步 | 接手B | 接手C-correct | 接手C-other | 接手C-wrong |
| --- | --- | --- | --- | --- | --- | --- |
| Object14 ketchup→basket | B | 25 | 45 | 39 | 38 | 43 |
| Object14 | B | 50 | 45 | 38 | 34 | 38 |
| Object14 | C-correct | 25 | 46 | 44 | 46 | 43 |
| Object14 | C-correct | 50 | 42 | 42 | 45 | 44 |
| Goal21 bowl→stove | B | 25 | 44 | 40 | 37 | 44 |
| Goal21 | B | 50 | 43 | 38 | 37 | 42 |
| Goal21 | C-correct | 25 | 34 | 31 | 25 | 30 |
| Goal21 | C-correct | 50 | 27 | 32 | 27 | 32 |

**已支持的局部因果事实。** Goal固定接手策略p时，换B前段的点估计均更高，25步四followers的增加为10/9/12/14，
50步为16/6/10/10。以B接手为例，C前段→B前段在25步为34→44（保留32、得12、失2），50步为27→43
（保留24、得19、失3）；描述性95%区间分别为[6,34]pp、[16,48]pp。
C接手的对应增加9和6，区间[0,34]pp、[−6,30]pp，不能把每格方向相同写成每格均已统计确认。
给B更不利的C历史也会伤害B，反对仅把Goal损失归因于C后续执行函数普遍较弱。
Object没有B历史的普遍优势；C-other接手在C前段下比B前段多8/11，区间[4,30]pp/[8,38]pp。
保持与获取依赖任务和前段/接手组合，不能整体压小C或将B前段当通用部署修复。

**条件与行为边界。** 固定历史下八组C-correct/C-wrong差值的95%区间全部跨0；这削弱所测接手状态上
“correct必然比wrong更差”的强解释，不证明视频无作用或两者等价。没有wrong前段锚，不能把旧自然rollout的11分差
作严格中介分解。C-other在相同Goal历史下仍有较低点估计，条件效应及成功集合变化继续保留。
Goal的C-correct接手：B25前段中48条曾移动目标、3条干扰物先持续位移；C25前段中对应39与11。
Object相同两格各50条都曾移动目标，仍分别失败11/6条，反对以错误对象统一解释两个任务。
持续位移不是抓持或语义判定。固定state0/25中既有接手改变结果，也有全部成功；不由单张图判因果。

**新的定位线索。** 后验几何复算覆盖每task全部50初态：第25步B/C前段下各物体body position相同，
EEF位置差的中位数Goal为5.38cm、Object为7.89cm。Goal的EEF到目标body中心水平距离均值B/C为4.38/6.16cm，
EEF相对目标body高度为20.77/17.16cm；Object对应14.10/9.93cm、21.89/16.62cm。
C在两个任务均较低，但水平接近在Goal更远、Object更近；因此水平命令偏置与下降/对齐协调成为可区分解释。
原controller限幅/缩放后的前25步累计z增量命令，两个任务均为50/50初态C比B更负；这不是实际末端位移的估计。
body中心不是抓取点，位置相同不证明姿态、速度或所有物理子步接触相同；这些几何量尚不是已验证中介。
代码和逐state数值在`coordination/approach_geometry.{py,json}`。

**不能遗漏的执行变化。** 新纯策略接手也改变了成功集合：Goal C旧25→新31/32，B旧42→44/43；
Object C旧46→44/42、B仍45但集合有交换。因此使用新分支内部对照裁决，不能把旧25→B前段后的40全算作干预净收益。
两任务系后验选择、单training seed；50-state bootstrap只给固定任务描述性不确定性，未作多重比较校正，不外推到全40任务。

裁决：关闭本批，保留早期机器人状态的局部因果作用；尚未解释Writer为何产生偏置，亦未证明有益视频增量或统一修复。
历史E32空间信用的局部正效应及Goal非通过、9月14日异质行为反例继续有效。
下一项登记[接近阶段动作通道干预](docs/designs/approach_channel_causality_design.md)：只在前25步做x/y来源与z来源2×2干预，
加完整B参照，验证水平接近、下降协调与其它通道的竞争预测；不是训练配方或新部署方法。

## 137. 动作通道干预完成：Goal的水平接近具有局部因果作用，单独抬高/减缓下降不是修复（2026-09-24）

Study：`/data0/user/ymdai/ember_runs/approach_channel_causality_20260924`，正式实现`6a9ea40c`。
14面板1000唯一分支、48 worker exit0、500组同prefix双接手历史、40固定双相机full cases齐全；无训练或新expert标签读取。
主讨论独立从原14份results核对全部行、模型/bank/teacher与RNG，并从原NPZ/contact重算命令来源、连续几何、
接触与持续位移；复算全部32个预定义配对对比、R/G/L和20000次联合bootstrap。与原分析一致。
前25步命令与两原donor逐通道一致；保存state误差和同prefix完整历史差异最大0，控制步采样未观察到前25步机器人-物体接触。
这不能排除采样间物理子步接触。代码/复算在`coordination/completed_recheck.{py,json}`及`geometry_recomputed.jsonl`。
40固定case四张拼图均已查看，来自原保存图像，无新policy或simulator调用。

每格50初态成功数；C为C_correct接手。这是诊断分支，不是部署分数。

| 任务/接手 | 纯C | B_xy+C其余 | B_z+C其余 | B_xyz+C旋转夹爪 | 纯B |
| --- | --- | --- | --- | --- | --- |
| Object14 / B | 43 | 43 | 46 | 47 | 47 |
| Object14 / C | 46 | 40 | 38 | 37 | 39 |
| Goal21 / B | 34 | 37 | 29 | 38 | 45 |
| Goal21 / C | 27 | 40 | 19 | 40 | 40 |

**主读出支持的机制范围。** Goal固定C接手，仅换x/y保留23、得17、失4，净+26pp，描述性95%CI[10,42]pp。
实际末端到目标body中心水平距离在44/50初态缩小，均值减少1.72cm，而高度均值只增加0.046cm。
目标持续位移37→49、干扰物持续位移16→1，观测到目标接触42→49、干扰物接触16→2。
这些相互一致的读数支持：本批前段水平命令改变接近状态，进而改善与目标/干扰物的接触和后续结果。
它不单独证明“语义识别错误”，也没有严格估计几何量或接触量各自的中介效应。

只换z确实在50/50初态抬高末端，均值+3.65cm，水平距离均值仅+0.015cm；有观测目标接触者的接触时点中位数39→43。
但成功27→19，保留16、得3、失11，净−16pp，CI[−30,−2]pp；目标位移37→32，干扰物位移16→19。
因此不是“干预没改变下降”，而是其未兑现H_z强版本的改善预测。接触时间的条件中位数受接触样本集合变化影响，
不能作为纯时间因果效应。联合xyz同为40，较xy净0、CI[−14,14]pp且6得6失；相同总分不等价。
交互I=+16pp，CI[−4,36]pp，未确认正交互；“必须同时替换两者才能改善”的强H_coord也不成立于主读出。

**反例和其余作用。** Object的C接手只换xy净−12pp、CI[−26,2]pp；只换z净−16pp、CI[−30,−2]pp；
联合xyz净−18pp、CI[−32,−4]pp。xy替换在48/50初态使水平距离增大，均值+4.16cm；z抬高均值+5.23cm。
纯C与联合xyz均50/50曾使正确目标持续位移，最终失败却分别4/13，不能由错误对象单独解释。
同前段下B接手结果不同：Goal完整B比xyz多7成功，CI[2,26]pp，Object并无相同强证据。
完整B还同时改变旋转/夹爪命令及动力学交互，这一差额不能单独归因其中某个通道。

**执行与外推边界。** 新纯前段相对前批有正常成功集合变化：Goal C→C从31到27（25保留/2得/6失），
B→C仍40但6得6失；Object C→C从44到46。全部8组差异保留在`analysis/pure_vs_prior.json`，
主要裁决只使用本批五臂内部对照。单training seed、两后验固定task、未作多重比较校正，不外推到全40任务。
前段重放是实际动作命令干预，不是新的合法EMBER策略；接触/位移不是抓持或语义标签。

裁决：关闭本批；局部支持H_xy，削弱单独改变下降即可修复及必须联合替换的强解释，保留其它通道和后续能力缺口。
不据此加时序loss、整体缩动作、用B固定前段部署或宣布统一根因。能力保持与有益视频增量尚未修复。
下一项[同状态交叉视频动作函数诊断](docs/designs/crossed_video_action_field_design.md)固定实际query与C@420，
交叉全部同task正确视频bank，以共有函数差异/视频条件变化的分解向上定位；本批动作因果证据不由离线读数替代。

## 138. 同状态全交叉：早期C/B函数差异主要为正确视频池共有分量，不能据此宣布视频无用（2026-09-24）

Study：`/data0/user/ymdai/ember_runs/crossed_video_action_field_20260924`，正式实现`2cfd1da9`。
300唯一query来自两任务各50初态的原C前段0/10/20步；15000正确条件预测加B/Source/封存wrong各300，合计15900次真实10-flow。
四worker exit0，pilot53计入；无训练、新expert标签、Writer物化或新闭环。首次4卡准入被拒时零预测启动，随后3卡完成，原拒绝保留。

主讨论从300份原预测NPZ、300份query、100条原C轨迹和原bank映射独立复核身份/噪声/8D重放/OSC变换，
再重算所有登记主分量及其20000次联合bootstrap。原state最大差0；OSC独立重算最大差2.24e−9。
原件与复算在`coordination/completed_recheck.{py,json}`；不是仅检查Sol的完成标记。

主要读数为前5动作的scaled OSC xy命令（不是实际位移）。T为C相对B差异平方量，M为C正确池均值与B之差，V为池内方差，T=M+V。

| task / control step | V/T | 描述性95%区间 | 池内RMS命令 / 共有差异RMS命令（mm） |
| --- | --- | --- | --- |
| Object14 / 0 | 0.113% | 0.095–0.137% | 0.355 / 10.553 |
| Object14 / 10 | 0.183% | 0.122–0.287% | 0.341 / 7.962 |
| Object14 / 20 | 0.388% | 0.274–0.593% | 0.328 / 5.249 |
| Goal21 / 0 | 3.532% | 3.163–3.906% | 1.840 / 9.616 |
| Goal21 / 10 | 1.419% | 1.174–1.721% | 0.753 / 6.277 |
| Goal21 / 20 | 3.735% | 2.354–6.054% | 0.889 / 4.513 |

六切片M−V区间均为正，支持本窗口H_shared相对“主要由哪条正确视频造成C/B差异”的强H_condition。
Goal t20的视频变化更多表现为query×video交互（约占V的71.4%），不能概括成一个固定视频偏移。
但这是函数平方量分解，不是闭环失败归因比例；低V允许利用视频共有内容，小动作差也可跨过接触边界。

补充事后描述性检查：未clip环境动作的V/T与主读数相近，Goal t10为1.581%，其余基本相同；
Object三个时间没有xy限幅，Goal限幅比例0.016%/4.216%/0。削弱“仅限幅把视频变化抹平”的解释。
各正确视频沿B−C均值方向的最大投影，在Object三切片及Goal前两切片均未到B；Goal t20有10/50个query可达到。
这只是固定池范围检查，没有挑选或部署最优video，也不证明其它方向的变化没有行为价值。

**wrong与Source不能遗漏。** 换wrong仍产生1.22–2.96mm RMS xy变化，大于或接近正确池内变化；
Goal wrong朝B−C均值方向的逐state投影均值为.153/.398/.493，但这不是有益视频证据，也不能把B当expert。
原四臂配对行中，Goal Source40、B42、C25：C对Source保留22、得3、失18；Object Source0、B45、C46，C新增46。
因此两个任务分别突出保持损害与新能力获得。当前Source在早期函数上靠近B不等于它有相同完整能力，不能统一回退Source。
同状态朝目标body中心的C命令投影在早期还更大，亦说明“更朝body中心”这个几何proxy不能直接当正确动作监督。

**不重复旧解释。** 历史Core/P交换、静态末P边界、自由A/B与任务梯度隔离已有明确作用及负例（§119/127–129）；
本批未识别某模块唯一失败，也没有理由只改时序loss、扩rank、默认增加视频数或重做同样消融。
额外CPU算子分析使用真实action_out矩阵ΔW及source读出W0：ΔW在W0行空间的真实7维Frobenius能量均值，
Object/Goal约2.270%/2.273%。这是参数空间描述，非实际hidden加权贡献；没有支持“单一动作倍率过大”的结论。
原件`coordination/output_operator_geometry.json`，与预登记主统计分开。

裁决：关闭本批。开始[动作读出与内部适配因果分解](docs/designs/readout_realization_causality_design.md)，
固定C的最终action_out与其余37个适配，四格开/关；用精确flow读出分解及400配对诊断同时检查Goal保持与Object获得。
这比继续动作轴扫描更接近参数怎样改变控制的计算机制；尚不等于已定位学习根因或选定架构修复。

## 139. 最终读出开关未兑现主要修复预测；内部适配实现大部分早期变化，但学习根因仍未识别（2026-09-24）

Study：`/data0/user/ymdai/ember_runs/readout_realization_causality_20260924`，正式实现`ed2df051`，
clean detached树`/data1/user/ymdai/projects/EMBER-readout-realization-formal`。
按事前[设计](docs/designs/readout_realization_causality_design.md)完成300真实query×四格=1200预测及400配对闭环，
16固定双相机full cases齐全；五个预测worker正常完成、八个evaluator worker exit0。无训练、新expert标签或官方Val/Test。

主讨论独立读取300预测NPZ、400原始闭环行及对应连续对象/EEF/动作/谓词trace与接触文件，核对初态/controller/
policy RNG/condition配对、两任务各50teacher无放回，再复算全部12个成功对比和36个xy函数对比及20000次联合bootstrap。
预测Euler重建与直接/反馈分解的float64复算最大余项6.68e−7/7.87e−7；原FP32汇总最大8.35e−7，均远低于登记容差。
关闭读出组的实际velocity delta为0；预定两任务state0/25共四条件的八份派生bank逐76个A/B核对通过。
16固定case的四张双相机拼图已查看。独立检查只读原件，代码/结果在`coordination/main_recheck.{py,json}`、
`mask_fixed_cases_recheck.json`及`fixed_cases_*`，不是仅接受Sol的completion。

**四格行为。** R为除最终action_out外的37个LoRA，O为最终action_out LoRA；各格每任务50初态。

| task | 00 Source | 01 仅O | 10 仅R | 11 完整C |
| --- | --- | --- | --- | --- |
| Object14 | 0 | 0 | 41 | 46 |
| Goal21 | 35 | 33 | 30 | 28 |

去掉O即10−11：Object保留40、得1、失6，净−5，描述性95%区间[−10,0]例；Goal保留24、得6、失4，
净+2，区间[−4,+8]例。Goal的+2不是稳定收益，也不是没有交换。相对同批Source，Goal仅O为27保留/6得/8失，
仅R为24/6/11，完整C为22/6/13；Object仅R独立获得41例，完整C获得46例。
成功率交互Object+10pp、Goal0，区间分别[0,+20]pp和[−20,+20]pp；没有以不显著证明两组独立或等价。
Goal仅R−Source为−5，区间[−13,+3]例，完整C−Source为−7，区间[−15,+1]例；两任务50states不足以独立确认总体保持损害大小。

**函数预测比两例净差更有区分力。** 以下均为真实十步flow后，前5×xy的scaled OSC命令RMS，单位mm，不是实际EEF位移。

| task / control step | 去掉O的变化：10−11 | 内部R的作用：10−00 |
| --- | --- | --- |
| Object14 / 0 | 0.263 | 15.778 |
| Object14 / 10 | 0.198 | 20.405 |
| Object14 / 20 | 0.166 | 12.630 |
| Goal21 / 0 | 0.087 | 14.219 |
| Goal21 / 10 | 0.057 | 8.757 |
| Goal21 / 20 | 0.059 | 4.852 |

Goal三个时间，移除O朝同query的旧B−新11方向的逐state投影均值分别−.00248/+.00106/−.000057，
描述性区间分别[−.00413,−.00081]/[−.00051,.00269]/[−.00400,.00395]。
事前H_O要求正向函数变化和闭环改善；初态方向反而为负，后两时间也未确认正向，闭环净差区间跨0。
因此不支持“最后读出是主要不利共同偏移、移除即可修复”的具体版本。B仍只是冻结方向参照，不是expert；
此处同query差异不能替代前批各自自然轨迹上保存B命令的因果作用。

**数学解释与范围。** 用本批真实flow轨迹有
`F11−F10 = sum_k dt*DeltaW_o*h11_k + sum_k dt*W0*(h11_k−h10_k)`，`dt=−0.1`。
第一项为直接读出，第二项为改变读出后经flow状态反馈的作用；两个项在规范化空间相加重建总干预。
Goal初态直接/反馈的signed xy均值，仿射反归一化再按OSC比例转换后，分别约[−.1148,−.0101]mm与[+.0866,−.0171]mm，
说明只看直接项会错估完整作用。它们没有分别clip；实际限幅后的总命令另由原预测计算。
“内部适配改变进入读出的计算响应”能解释作用位置；仍没有说明Writer为什么学出该映射、视频是否提供了有用知识，
也不允许由这些早期小量宣布O在后期或其它通道无用。Object去掉O确实有1得6失，保留该反例。

H_R在早期函数作用位置上得到支持；其“主要造成Goal行为损害”的强版本仍受上述闭环区间和训练混杂限制。
H_coupled中“单独内部适配不能获得Object能力”的强版本被41/50反驳；但不能据此抹去联合后的额外能力或一般交互。
这些都是冻结推理干预，不是分别训练两组；不从冻结mask推导分阶段训练或冻结组能得到同样结果。

**新旧结果边界。** 本批Goal Source35相对原40为保留33、得2、失7；C28相对原25为22/6/3。
Object C总数仍46但44/2/2，Source仍0。只用本批配对作主比较，不把原Source40代入新10=30以放大差额；
正常数值/动力学变化保留，不追逐逐bit复现。只有两后验固定任务、一个training seed，区间是描述性的。

**跨轮裁决。** §137定位了Goal早期水平命令的行为作用；§138说明C/B早期差异主要为正确视频池共有分量；
本批将大部分早期函数作用定位到最终读出之前。这条链比现象排名更窄，但还没有走到训练根因。
不能说视频完全没用，也不能说已有有益视频增量。旧Core/P正反例、末P边界修正阴性、private Writer不输freeAB、
任务隔离和H54缩头阴性继续约束后继；没有依据按内部层编号继续扫，或由Goal单任务统一压低全部适配。
关闭本批，不采纳action_out冻结，不宣布修复；主讨论转向任务关系支持与功能学习信用的判别设计，见task_plan第七阶段。
该次裁决时新的实现/训练合同尚未形成，没有向Sol追加任务；后续登记见§140，长期研究授权持续有效。

## 140. 任务关系支持可形成受控的2×2干预；图的可辨识性是待测模型，不是已证实的训练根因（2026-09-24）

本条是metadata与代数审计，**不是新训练或闭环结果**。依据§135–139的异质获得/保持、共有函数偏移和读出假设阴性，
后继改为干预Writer学习数据的具体关系支持。历史meta73/target18（§68）同时改变权重/总监督量，已明确约束“只加任务”解释。

注册`relational_support_causality_20260924`：保持诊断held8、每臂28任务、1260更新、模型/Source/优化器和21+7监督预算。
所有新臂共同移除原fit28的43/56/95/96，增加58平底锅→炉面、83白碗→双层架顶面；另以两因子替换76/77及80/81。
因子i为白碗“盘上/盘右桌面region”，j为平底锅“双层架顶面/架内region”。四个fresh视频C配S00/S01/S10/S11，
两个fresh语言B配端点S00/S11；共同背景变化只在新批内部比较，不拿旧C当新S00。

已检查canonical source manifest、完整BDDL审计及实际安装的四份完整BDDL：六个候选均在Source71，非目标40完整等价任务。
替换对76/77和80/81都只有language/goal块不同，regions、fixtures、objects、init定义逐结构相同；
不等于HDF5演示、saved init states或动作相同。原字面“cabinet”实际包括不同装置，不能把wooden_cabinet和wooden_two_layer_shelf合并。
新protocol保持官方split原样；canonical loader验证58任务/42Train/8Val/8Test，训练每臂另限fit28且排除诊断held8。
全程零新HDF5数组解码、零模型forward、零新hash；沿用来源的现有身份元数据。

在明确的`c(o,d)=u_o+v_d`低阶控制表示假设下，监督关系设计矩阵每行是`e_o+e_d`。
Goal21的黑碗→炉面行，在S00/S01/S10不属于训练目标关系行空间，在S11属于；CPU投影残差约.601/.527/.584/3.6e−14。
准确路径为黑碗→盘子−白碗→盘子＋白碗→双层架顶面−平底锅→架顶面＋平底锅→炉面。
这是矩阵行的等式，**不对LoRA/视频/动作作加减**，不证明实际Writer内部加性或原任务不能泛化。
Source已经见过全部新增任务，Spatial7已有黑碗从炉面移出的初始关系/反向操作；语言与原生先验可以超出此目标图模型。

事前主预测为共同1260节点Goal21正确视频的正交互`C11−C10−C01+C00`及C11对C00的绝对收益，other也应兑现。
预先区分两个一般主效应、语言与视频共同的数据收益、支持操作尚未习得、非线性场景/优化交互和成功率上界；
阳性不直接确诊统一根因，阴性只否定本窗口的具体支持修复预测。完整八任务/相邻节点保持继续约束Goal局部解释。
wrong仅冻结选点后作解释，不能用下降制造有益视频差值；不跑时序controls或新增官方Val/Test。

审计原件：`configs/relational_support_causality_v1/partition_audit.json`，含所有fit目标边、路径、full spec与来源；
设计：[任务关系支持的学习干预](docs/designs/relational_support_causality_design.md)，完整规模/参数/评测在同目录spec。
主讨论本批已形成具体合同并接续派发，不再把“批次完成、下一设计未定稿”当作停止自身工作的理由。

## 141. 关系支持诊断的完整曲线成本超过核心辨识需求；在产生结果前收缩为固定终点及相邻点（2026-09-24）

这是设计效率裁决，不是新模型结果。Owner质疑20–26小时剩余成本是否对应足够信息量后，核对原21868条中
16704条用于六臂六节点held400/seen64，而主假说一直只在1260检验；Goal21仍仅50配对状态、一个训练seed。
早期曲线能说明学习过程，但不能替代独立训练重复或自动消除非线性优化/场景竞争解释。
六臂仍有必要：四C辨别两个关系主效应与交互，两B辨别一般数据收益；support检查操作是否习得，
完整held400/seen64约束局部修复的能力代价，other/wrong检查正确教学收益，均保留。

14:38 UTC确认本批evaluation中run_contract/results/launcher_completion均为0后，事前固定所有臂1260为唯一报告模型，
1050只作相邻保持。取消210/420/630/840评测及原最佳点搜索/非1260额外other，共9932闭环：
6×2×464＋2400 controls＋1200 support＋764 Source。3300封存query预测、82固定cases及原被动采集要求不变。
不改变六臂数据、初始化、训练步数/损失/拓扑或冻结7dc运行树；12个训练checkpoint均保留。
少11936条（54.6%），换取放弃早期曲线/峰值结论；不能将本修订说成由好坏分数触发的早停。
实际评测只从`evaluation.executed_updates`取1050/1260，原冻结训练节点元数据不再作为评测launch清单。
实现与派发状态见progress；不能把设计收缩冒称执行器已经修好，更不能把负结果自动判成架构根因。

Owner随后明确：少量评测不足时，有必要即可随后补测，并要求吸收为后续经验。
因此效率原则是先获得足以改变下一步判断的最小证据，再按具体缺口追加；并非把9932变成永久上限。
补哪些证据取决于歧义来源：状态不确定性、训练seed方差、视频作用或学习阶段分别需要相应样本/重复/对照/节点，
不能用反复评同一批早期节点代替所有独立证据。先写清追加理由、数量和停止条件，保留原始主比较与所有既有结果。

## 142. 先验证核心关系假说再决定能力评测扩展；1500条阶段合同（2026-09-25）

仍无新闭环结果。Owner进一步质疑9932全套的必要性，主讨论承认前次只是缩曲线，未充分拆开核心诊断与能力验证；
随后Owner明确要求据此指导Sol。16:39 UTC（北京时间00:39）确认没有正式评测面板后登记1500条第一阶段。
六臂训练和固定1260保持；Goal21与Object14各50state/臂共600，Goal21四C other200，新增四support各20/臂共480，
新Source两目标100与六support各20共120，合计18面板1500行；原病例交集66例，主问题50-state/video映射未缩减。
1050、剩余held/seen、wrong和3300预测暂缓，不预物化。support20截取原50映射，后续必要时补剩余30且不重跑已完成行。
本阶段能初步区分预期交互、一般数据收益和新增操作未习得；support20的中间结果仍可不确定，也不证明全部旧支持关系已习得。
不能据此声称完整保持、相邻稳定性或视频因果修复。完成后主讨论按实际证据缺口决定补哪些样本/对照，执行者不自动补矩阵。
当前相对9932减少约84.9%的自动闭环工作；同时允许已就绪臂在剩余训练期间先评，保持原资源与信息墙。

## 143. 关系支持第一阶段完成：正交互没有兑现绝对修复，数据变化伴随Goal接近和保持损害（2026-09-25）

根`/data0/user/ymdai/ember_runs/relational_support_causality_20260924`。六臂均完成固定1260更新，训练7dc95edb；
18面板1500有效闭环、66full、1500条T+1 trace完成。保留C_S01 correct100的E2原件，其余1400来自E3=`385ae992`；
两有效E2 bank及other复用来源按登记例外保留，未重跑100行。早期support中止尝试单列，不能从0有效行推断0控制步。
主讨论独立读取全部results、连续NPZ、合同与worker回执，核对唯一行、条件/噪声配对、50视频无放回、1260身份、
T+1/动作有限值、终态谓词和成功一致；重新计算29个预定对比、44组R/G/L及20,000次state-paired bootstrap，均与执行原件一致。
代码与结果为`coordination/main_stage1_recheck.{py,json}`。没有补测held400、其它节点/seen/wrong或3300预测。

| 固定1260模型 | Object14 correct /50 | Goal21 correct /50 | Goal21 other /50 | 各自四支持任务 /80 |
| --- | ---: | ---: | ---: | ---: |
| Source | 0 | 38 | — | 六任务103/120，不与不同任务集合直接作合计差 |
| B_S00 | 40 | 43 | — | 76 |
| B_S11 | 39 | 37 | — | 78 |
| C_S00 | 42 | 36 | 40 | 74 |
| C_S01 | 38 | 37 | 37 | 76 |
| C_S10 | 38 | 26 | 29 | 77 |
| C_S11 | 40 | 35 | 33 | 77 |

**事前关系修复预测未兑现。** Goal correct交互`C11−C10−C01+C00`为+16pp，95%CI[-8,40]；
other为+14pp，CI[-6,34]。但C11对C00绝对变化为correct−2pp、CI[-18,12]，other−14pp、CI[-28,0]。
因此不把正点交互当作有效组合或修复：i替换在j=0时使correct−20pp（CI[-36,-4]）、other−22pp（CI[-36,-8]），
在j=1时为−4/−8pp；j主要在较差的C10背景上部分恢复。没有证明精确零效应，也没有证明所有关系支持无效。
B11−B00的Goal为−12pp；correct差中差+10pp、CI[-8,28]主要受B退步影响，other差中差为−2pp。
不能以语言参照下降制造视频增量。C11 correct35、other33，均未超过同池B37；C00 correct36/other40也未超过B43。

**新增支持操作已有较好表现，不能把全部阴性归为根本没学会。** C11在58/76/81/83分别20/17/20/20，
C10在58/76/80/83为20/19/18/20。最弱是C00的77为15/20，仍非近乎零成功。
Source六任务为19/15/12/19/18/20；高支持分包含原有能力保持，不能声称这些关系都是Writer新学到的。
原路径中的旧共同黑碗→盘子边未单独验收，20state也不是操作充分掌握证明。因此削弱的是本预算/模型下
“仅补齐这两处监督边即可修复Goal”的预测，不是可组合控制表示的数学可能性。

**能力保持仍有实质交换。** Goal相对Source，C00/C01/C10/C11的R/G/L为28/8/10、30/7/8、23/3/15、29/6/9，
B00/B11为36/7/2、30/7/8。C00→C11为28保留/7得/8失，总分近似并非同策略。
C00→C10则21保留/5得/15失；同数据数量和训练预算下，一条支持任务替换已经改变了目标控制与保持。
Object各C仍新增38–42个Source失败条件，不能把Goal损害解释为整个模型没有获取能力；尚无其它六任务或相邻保持结论。

**原件支持的后验行为线索。** 全50状态的C00/C10/C11，25控制步EEF到Goal目标body中心水平距离均值为
4.62/7.05/5.36cm，相对高度19.52/18.33/19.47cm。按>2cm且连续5控制步计算，目标持续位移45/35/41条，
干扰物先持续位移5/16/11条；1cm和3cm阈值方向保持。B00相应目标48、干扰物先动2。
这些是身体几何与运动描述，不等于已识别抓持或语义内部机制，也不估计中介贡献百分比。
Object的C00/C10/C11目标持续位移为48/50/50，移动后仍失败6/12/10，继续反对用错误对象统一解释两任务。
原件为`coordination/main_stage1_geometry_rows.jsonl`；固定Goal state0/25的Source/B00/C00/C10/C11十例双相机拼图已查看，
十例均成功，不能用它们代表总体失败分布；连续全行统计与固定案例分开解释。

**裁决与下一步。** 关闭本批第一阶段，不自动补全原矩阵、延长训练或为找正交互扫seed。
这轮得到的是一个可复核的数据干预效应与未兑现的修复预测，仍没有经过验证的统一根因或方法。
下一批[原生读取适配与其余Writer的冻结交叉](docs/designs/native_reader_transfer_causality_design.md)，
只问C00→C10的数据效应怎样传递。两任务四格400闭环、无训练，区分三组Native Meta变化、其余Writer变化及协同适应。
旧§44–46已证明不同模块与交互都可能参与，不能重述为某端一定有害；新批利用固定数据替换端点和原生坐标边界，
不据混合涨分直接冻结模块，也不当部署checkpoint融合方案。具体派发/实现状态见progress。

## 144. 原生读取交叉完成：后续条件映射传递主要损害，读取变化在Goal上部分补偿（2026-09-25）

研究根`/data0/user/ymdai/ember_runs/native_reader_transfer_causality_20260925`，正式实现`edca1a35`；
C00/C10父训练7dc95edb，两个self bank只读复用原E3，两个混合bank各100次完整Writer生成。
400新闭环、16full、400条T+1 trace和16预定对比完整。主讨论检查实际N/W加载路径及self真实动作复现，
从全部results、轨迹和连续NPZ独立重算成功集合、12组R/G/L、16对比/20,000次bootstrap、100首轮函数分解及全行几何，均一致。
四格初始8D state、物体/EEF/夹爪差异max=0；首轮动作与实际控制步一致，成功终止后的未执行chunk尾部按实际步数截断核验。
原件`analysis/`，独立核验`coordination/main_recheck.{py,json}`。无新梯度、expert标签、官方Val/Test或部署成绩。

| 新批self及交叉 | Object14 /50 | Goal21 /50 | Goal非目标物先持续位移 /50 |
| --- | ---: | ---: | ---: |
| N0_W0 | 45 | 39 | 5 |
| N1_W0 | 42 | 38 | 4 |
| N0_W1 | 36 | 19 | 26 |
| N1_W1 | 37 | 28 | 18 |

N只指三组Text/VL/Action Meta；W还包括projection、patch grounding、Core、Procedure、Compiler和heads，不能称为仅输出头。
Goal换W在N0下−40pp，95%CI[-56,-24]，在N1下−20pp，CI[-36,-4]；N在W0下−2pp，CI[-14,10]，
在W1下+18pp，CI[6,30]；交互+20pp，CI[4,36]。N0W1→N1W1为18保留/10得/1失。
Object换W在N0/N1下分别−18/−10pp，区间[-32,-6]/[-24,4]；换N为−6/+2pp，未见同样明确的补偿。
因此本例不支持“原生读取变化主要致害”，支持后续条件映射的变化足以传递明显损害，同时保留协同适应；
没有证明某个W子模块坏、N普遍无害、冻结N有益或混合模型可部署。

前5步环境xy的有限精确分解`D=F11−F00=E_N+E_W`重算余项0。
Object沿D投影N/W=.03292/.96708；Goal为−.10428/1.10428。负投影与Goal读取补偿方向相符；
不能称为负10%/110%的失败归因，闭环与内部向量不是同一对象。
旧stage1与新self分开：Object C00 42→45、C10 38→37；Goal C00 36→39、C10 26→28。
Goal两self仍有39→28的退步，但所有主效应只用新批内部四格，不拼接旧分数。

**后验定位进一步具体化。** Goal实际语言是`put the bowl on the stove`，目标为黑碗→炉面。
>2cm且连续5控制步的非目标物先动，前三格分别全是plate 5/4/26；N1W1为plate17、cream_cheese1。
这里的plate确实是该任务干扰物，不能误说成教学白碗、该任务目标容器或内部语义误判。
Goal的25步EEF到目标水平距离四格为4.62/4.27/7.55/7.05cm，目标持续位移45/47/29/34。
固定state0/25八例双相机查看均成功，只说明这些固定案例，不代表失败分布。

用已保存动作按实际OSC裁剪/缩放，沿初态黑碗body中心→plate中心单位向量投影：
W有限分量在50/50状态为正，前5命令平均+0.009407m/命令，state bootstrap CI[.008391,.010483]；
N为−0.001723m/命令，49/50为负，CI[−.001941,−.001509]。
这是后验的**控制目标增量**方向，不是实测末端位移、抓持或完整中介比例；body中心不是抓取点。
原件`coordination/goal_bowl_plate_projection.json`，下一批可将这一方向作为事前功能预测。

**学习机制仍待区分。** 两个月的有限更新/模块混合（§44–46）、当前private Writer≥freeAB及缩head失败（§129）、
任务独立梯度不修复（§127）、有用旧P贡献（§118–119），共同反对“视频无信息”“头容量不够”“共享一律有害”的简单归因。
当前更具体候选是共享条件映射的更新把其它训练操作的功能偏置传到这个未参与梯度的条件；当前结果定位了传递端，尚未验证这条学习因果链。
原S00→S10同时删除77和加入76，不能仅据端点说76有害，也可能是77提供了保护，或较早共同适应形成的路径依赖。
下一批从同一C00@1155完整状态做28次有限更新，保留77/替换76/仅屏蔽77当前梯度三臂，
固定另27任务全部事件与optimizer，加入未更新父参照。它区分增添损害、删除保护和共同训练漂移，
并检验最初真实10-flow是否先出现向plate的偏移；不重开模块扫描，不自动fresh重训或补旧大矩阵。

## 145. 单项监督有限更新否定了预定的plate方向解释；删掉该项也没有同时保住两项能力（2026-09-25）

Study `/data0/user/ymdai/ember_runs/support_slot_credit_causality_20260925`，唯一正式实现`e58e8169`。
共同父C_S00@1155、三臂各28更新；KEEP77/SWAP76/DROP77的108共同事件完全一致，四slot按登记生效。
主讨论审阅实际梯度恢复/事件替换代码，并从400原始results、torch轨迹、NPZ连续trace及训练事件独立核对配对、
真实执行动作、终态谓词、12成功差和bootstrap；首次30预测从NPZ复算，128供体FM的query/noise对应。
核验脚本及结果为`coordination/main_recheck.{py,json}`；初态配对误差0，400trace/16full齐全。
KEEP/SWAP各3136非零信用queries，DROP3024；三臂均28个Adam更新，无裁剪。DROP没有清空其它累计梯度或历史动量。

| task，每格50 | 未更新P1155 | KEEP77 | SWAP76 | DROP77 |
| --- | ---: | ---: | ---: | ---: |
| Object14 | 46 | 41 | 43 | 39 |
| Goal21 | 40 | 42 | 45 | 48 |

Goal KEEP相对DROP净−12pp，95%描述性state配对区间[−22,−4]；DROP保留KEEP全部42例并新增6例。
Object KEEP相对DROP为+4pp[−4,12]；从KEEP到DROP保留38、新增1、丢失3。
SWAP相对DROP：Object+8pp[2,16]、Goal−6pp[−14,0]；Goal的DROP也保留SWAP全部45例。
相对父，DROP的Goal保留39/新增9/丢失1，Object为37/2/9；不能把Goal48/50称为联合修复。
区间为同50-state bootstrap20000/seed20260925、单训练seed、未校正多重比较，不是方法总体效果保证。

**原预测没有成立。** 第一个受干预更新1160后，新增76相对DROP的前5真实OSC xy沿黑碗→plate投影，
五个固定Goal初态全部为负：约−0.192/−1.066/−0.288/−0.435/−0.577 mm/命令；
KEEP−DROP为正/负混合。A−R=替换作用的恒等式独立复算通过。
末点50-state的SWAP−DROP平均仍为−0.882mm/命令，49/50为负；KEEP−DROP为−0.264mm/命令，38/50为负。
这些是动作命令的差，不是EEF实际移动。末点plate先于目标持续移动2cm为父3、KEEP2、SWAP1、DROP1；
目标移动为48/48/49/49。没有出现事前H_add预测的“76先把动作推向盘子、随后更多plate先动”。
H_remove要求KEEP提供闭环保护，本窗口Goal方向也与之相反。只能保留早期共同适应/历史路径等未识别解释，
不能用这些未检验解释挽救已失败的晚期局部预测，也不能外推76永远有益或77永远有害。

**方向代理也受到限制。** SWAP比DROP更远离plate，却少3个Goal成功；大部分轨迹已让正确的碗移动。
因此§137/144的早期接近因果事实仍成立，但plate方向不是完整成功的充分统计量，不应被直接写成训练目标。
Object各臂几乎全部让正确目标移动、没有其它物体先动，不能以统一错物体解释本轮损失。
供体随机tau/fullH FM各臂均有小幅变化，没有新增专属供体掌握或有益视频证据；不据loss反推根因。

旧§44的有限跨任务伤害、§127隔离所有其它任务反而丢收益、§129 private Writer不逊freeAB及缩头阴性继续有效。
本轮增加的是同一数据项作用的状态依赖反例，**尚未证明共享生成器的某个Jacobian性质、某层或优化器就是根因**。
既不继续删task，也不扩大当前28步窗口追逐支持预期的节点。本批关闭；保守GPU用量1.954小时、data0新增4.423GiB。

下一项先检验一个训练机制的计算前提：按真实任务划分的对称lookahead，是否比普通更新和混合任务分组改变跨任务信用，
以及这种变化能否传到独立query上的真实10-flow前缀。只登记七个独立单步case，不预先批准长训练或新闭环矩阵；
loss/梯度或演示动作改善均不能代替后续闭环与视频增量。具体合同见
[任务分组lookahead诊断](docs/designs/metatask_lookahead_credit_design.md)。

## 146. 任务分组lookahead没有提供超越普通更新的依据；弱对照与少数夹爪读数不能当作修复（2026-09-25）

Study `/data0/user/ymdai/ember_runs/metatask_lookahead_credit_20260925`，唯一正式实现`99c6491a`。
七个case分别恢复同一C_S00@1155，原1156..1162事件覆盖fit28一次，784原训练query；不是连续七步。
21个候选/28个虚拟状态是weights-only诊断资产。224bank、3584独立episode FM、448真实10-flow/112query及七份exit0齐全。
主讨论审阅分组mask、原噪声/权重、完整恢复及实际Adam路径，从全部rank原件与flow NPZ独立重算配对、
六个case-cluster bootstrap、有限函数恒等式，并从保存权重核对实际位移与虚拟步幅；
`coordination/main_recheck.{py,json}`保留源码和数值。未重跑模型、读新标签或生成新环境步。

正号表示误差更大；七case共同bootstrap20000/seed20260925，描述性95%区间：

| 指标 | TASK−BASE | MIX−BASE | TASK−MIX |
| --- | --- | --- | --- |
| 独立FM | +3.73e−5 [−4.34e−6,9.80e−5] | +1.06e−4 [2.49e−5,1.97e−4] | −6.87e−5 [−1.55e−4,7.17e−6] |
| 实际10-flow前5×7 MSE | +4.46e−5 [−5.36e−5,1.69e−4] | +9.55e−4 [−9.45e−6,2.76e−3] | −9.11e−4 [−2.60e−3,−2.83e−5] |

P/BASE/TASK/MIX前缀MSE为.104424/.100779/.100823/.101734。三种更新均改善父模型的平均读数，
TASK优于MIX不能代替TASK优于BASE。TASK−MIX总差的约94.0%来自task5/query2；此处BASE/TASK较MIX更大幅纠正了
父模型过早改变夹爪命令的预测。它保留在总表中，不删除“离群点”，但不能解释为普遍跨任务修复或真实抓取收益。
BASE的FM在17/28task下降、实际前缀MSE在16/28下降；其中8task两者方向相反，且FM有16query、flow有4query，
不将这一读数差直接归为同query梯度冲突或某个flow积分环节错误。

**干预确实改变了实际参数步。** TASK相对BASE的位移差为BASE步长的5.92%–17.09%，
TASK总步长为BASE的97.77%–99.94%；没有大幅统一缩小。MIX相对位移差12.44%–26.87%。
十四个TASK定向虚拟更新均降低本组训练loss，另一组7升/7降；独立episode本组平均FM变化−.000703，
另一组−.0000181。局部有益与有害迁移并存，不支持把所有共享信用叫作冲突。

**数值辨识限制。** g0与分组均值的范数相对余项TASK为.222%–1.088%，MIX为.677%–1.224%；
它不是Adam后动作误差的上界，不能与参数位移百分比直接相减，也不能断言所有微小差额都来自曲率。
原件未保存可直接形成零虚拟位移sham的全梯度，未定量分离重复VJP余项经Adam/flow传播的贡献。
但当前没有超越BASE的收益，追加精度实验不会为长训练提供已有正证据；不因此扩dtype、固定低吞吐batch或追逐bit一致。
恒等式按保存FP32输出用float64复算最大余项9.03e−17；这只验证有限代数，不消除模型计算的数值边界。

本批关闭，case核心计算计时1.0195 GPU-hours（不含初始化），data0新增3.294GiB。
执行者将该数标作保守GPU上界不准确；从launch合同建立至完结按两卡全占计算为1.405 GPU-hours，仍低于4小时限额。
计时边界复核保存在`coordination/resource_recheck.json`，没有为此重跑计算。未识别出值得延长的学习收益，不否定所有meta-learning、
不能外推七组/一个父节点为模型上限，也没有新的闭环、保持或有益视频结论。

**后继选择。** 不把当前代理改善当作继续更换loss的依据。下一项直接问当前共享图能否接收自己的真实成功信用：
八个合法训练task采集一次128条有界探索，形成共享回报梯度、其反向和同teacher原FM梯度的三个独立单步候选，
与父在独立初态和两条正确teacher上作最多256条闭环。参数步长固定为本批普通更新的RMS，不按回报调幅。
这检验可利用的行为信用，仍不能单独分开目标函数与expert/on-policy状态分布；正结果也须后续验证未见task和视频增量。
完整机制/反例/停止条件见[成功信用方向干预](docs/designs/return_credit_direction_design.md)，实际派发状态见progress。

## 147. 一次成功信用更新没有兑现方向性收益；稀疏信用与有限动作位移仍未分开（2026-09-25）

Study `/data0/user/ymdai/ember_runs/return_credit_direction_causality_20260925`。
原128采集/八bank来自dd2e00bc；按§6例外，完整梯度、三个独立候选、1536重放和256闭环统一4e3ade36。
主讨论审阅真实score、十步flow/完整Writer VJP、fresh SGD及数据路径，从32组和64评测面板原件重新核对
384条T+1/谓词、256条动作轨迹、16full及首态/噪声配对；六个R/G/L和20000次联合bootstrap全部复算一致。
保存权重验证实际三步范数均约.17076466，R/NEG反向、FM为loss下降方向，未活动buffer未变。
核验`coordination/main_recheck.{py,json}`；不存在把旧失败部分梯度或旧闭环分数混入本批的证据。

| 每task八条（四state×两teacher） | P | R | NEG | FM |
| --- | ---: | ---: | ---: | ---: |
| 2 | 8 | 8 | 7 | 6 |
| 5 | 4 | 4 | 5 | 4 |
| 12 | 4 | 5 | 0 | 1 |
| 17 | 6 | 0 | 0 | 6 |
| 22 | 7 | 6 | 8 | 8 |
| 25 | 0 | 0 | 0 | 0 |
| 34 | 5 | 3 | 6 | 4 |
| 37 | 1 | 0 | 0 | 0 |
| 合计/64 | 35 | 26 | 26 | 29 |

R−P为−9/64，task/state配对95%描述性区间[−34.375,+3.125]pp；保留24、新增2、丢失11。
NEG−P也−9/64，但保留22/新增4/丢失13；R和NEG各有6条对方失败的成功，不能把相同总分叫作相同动作或无方向作用。
FM−P为−6/64，[−25,+4.6875]pp。六个总对比区间都含零，不能声称总体必然退化；
但事前要求的R对P/FM的可迁移正收益及反向区分没有兑现，本批不支持启动长程RL。
两条teacher仍属有限训练task机制面板，不是视频必要性、未见task泛化或新的部署资格。

**信用估计的限制有直接证据。** 仅6/32组有LOO变化：task12/17/22各一组、37三组；
其余四task本批回报梯度为零。按初态偶/奇分组得到的完整梯度cosine为−.02686，task37自身为−.00284。
这是一份状态分组的单次读数，不能当总体SNR或独立重复seed的估计。
task37对总方向的内积贡献`<g37,g>/||g||²=.68625`，不是它造成68.6%失败的因果比例。
该事实支持继续区分采样方差/状态依赖，不能由一次无收益断言共享架构接收不了真实回报。

**同参数步幅不等于可比的局部动作步。** 同批评测首态的`odd=(mu_R-mu_NEG)/2`和
`even=(mu_R+mu_NEG)/2-mu_P`可直接从四格轨迹复算。task17前五步xyz的odd RMS=.12185、even=.02530，
说明它的两个失败方向已有较大的相反平移作用；首态夹爪改变量仅约.003–.005，不能归为共同早期夹爪翻转。
task12/37的even又不小；这些有限分解包含完整生成/flow/正常数值作用，不能唯一定位某层曲率。
后续保存query的候选重放有较大夹爪位移，但含重新物化及不同阶段状态，不据此宣布夹爪是闭环失败根因。
仍保留J_Sigma与J0、采集与评测state/video分布、一步非线性、稀疏信用的竞争解释。

**工程修正成立的范围。** E2两rank TF32=True，实际采集bank的512个decision重放RMS均0；
五个非零Writer条件重编译相对L2仍.002548–.003058，使用的是重建Jacobian，不称浮点严格无偏。
这解决了已验收的重放合同，未证明TF32单独解释全部旧差异；不追加精度矩阵。
有回执GPU计时独立相加3.339812小时；另.05小时是未精确计时诊断的预留，不能称已验证的占用上界。
study5601MiB、两formal树504MiB均在上限内。资源边界见`coordination/resource_recheck.json`。

本批关闭，不扩大采集、调步长、延长RL或按task重加权。§83的夹爪sign机制提示一个可计算的方差来源：
执行只看正负，Gaussian latent score仍给同一正负区间内的幅度分配随机信用。
CPU原件`coordination/main_gripper_score_{audit.json,rows.jsonl}`核对512保存decision均无采样sign翻转；
96个非零信用decision距边界最少9.5479sigma，夹爪幅度占加权输出score平方和3.7131%，
条件化后的该块平方和约为原块1.34e−43。联合翻转概率上界3.31e−21、条件score逐坐标上界6.39e−20，
不能把浮点概率1称为数学上严格1，也不能把动作端3.7%直接当完整Writer梯度的影响。
后继[冻结分解](docs/designs/return_score_conditioning_design.md)检验该分量经过完整Writer后是否实质改变方向；
它针对这次回报估计器，不是把两个月的监督/视频不足归结为夹爪，也不自动授权新闭环或优化候选。

## 148. 夹爪幅度的执行无关信用实质改变完整Writer方向，但没有消除两组初态的不一致（2026-09-25）

Study `/data0/user/ymdai/ember_runs/return_score_conditioning_20260925`，唯一实现d6348660。
只读复用§147的128采集、512decision、dd2实际bank与父1155；96个非零A输入、RAW/RB各一次共192真实十步flow，
五个活动task×parity组、11个显式零组、16份梯度向量完整。没有参数更新、候选、新环境步或新闭环分数。
主讨论审阅相关Gaussian条件score、实际bank VJP与同图Writer双反传，独立从全部512原始latent/mean/Q/M/A
及保存向量复算未改的30坐标、夹爪条件score边界、16组/总向量和几何读数，
`coordination/main_recheck.{py,json}`保存代码/结果。raw score相对重算余项最高3.83e−7，
各总/偶奇向量与组和相对余项<2e−9，主数值均与原分析一致；不是仅采信执行摘要。

| 固定读数 | 独立复算 |
| --- | ---: |
| `||g_RAW−g_RB||/||g_RAW||` | .587716 |
| RAW与RB cosine / 夹角 | .837657 / 33.106° |
| RAW重算相对旧E2差 / cosine | .001609 / .999998705 |
| 被移除分量与RAW的内积份额 | .116577 |
| 偶/奇初态cosine，RAW→RB | −.026864→−.024499 |
| task37对总方向平方的内积份额，RAW→RB | .686248→.713623 |
| 总范数，RAW→RB | 286.584665→302.242536 |

同一批数据/父/实际bank、只替换夹爪块的条件score，足以使完整参数方向发生实质变化。
RAW重算差约为干预改变量的.274%；没有依据把这次方向差主要解释为正常重算差异。
这反驳本批H_small，支持H_amplitude的“进入完整更新方向”部分。
但偶/奇一致性几乎未改善、task37占比未下降，H_amplitude的更强解释没有成立，H_other仍保留。
不能把58.8%叫作58.8%的总体方差/性能损失，也不能将内积份额称为某任务的失败比例。
这些范数/角度依赖参数化，尚未证明执行函数或闭环收益。

变化集中在task12/even（相对1.11094，cos .43217）和task37/even（.69665，cos .77921）；
task17/odd、22/even、37/odd分别为.02127/.04923/.02686，cos均>.9987。
这保留了不同state/task作用强度差异；不能据此删掉某个task或认定task17的旧损失已被解释。
总RB范数反而高5.46%，也不违背条件期望减少**总体条件方差**的定理：单次样本的范数不必下降。
Source冻结、三Meta活动、所有活动输入bank重放RMS0；Writer重建相对L2约.25–.31%的原边界继续保留。

GPU退出回执覆盖torchrun初始化至双rank退出260.816秒，两卡合计.144898 GPU-hours；
CPU阶段仅有退出观察记录、未计时，不纳入GPU占用。data0 740MiB，开发+formal代码约505MiB，在限内。
本批关闭，已经确认的是一个估计器的计算机制，尚未验证它是旧闭环下降的原因，亦不是监督/视频问题的统一解释。

下一项[96条闭环补验](docs/designs/return_score_update_design.md)：同父、封存RAW/RB梯度、相同SGD系数，
比较P/RAW/RB在原八合法task、states32/33、teacher46/47的成功与保持。保持优化器系数而非单独归一RB，
不按新成绩调幅；无新梯度/回报采集，使用闭环首轮已有输出检查函数作用，不另铺预测矩阵。
若无相对RAW的实际收益，不沿这一修正继续训练；其余states或更大评测须另论信息价值，不自动追加。

## 149. 去除夹爪幅度信用后新增两条成功，但没有修复父能力丢失（2026-09-25）

Study `/data0/user/ymdai/ember_runs/return_score_update_causality_20260925`，统一正式实现34ea27bd。
父P、封存RAW/RB梯度的两个独立fresh SGD候选、48新bank、96闭环、12full及全T+1 trace齐全。
主讨论审阅生成/更新/评测路径，从每行completion、轨迹、连续trace和候选权重独立复算三对比及联合bootstrap、
首轮函数与实际步长，核验`coordination/main_recheck.{py,json}`；首态配对和实际动作对trace的最大差均0。
545可学张量、77固定buffer，未继承Adam；同一alpha=.0005958611283018859，RAW/RB步长约.17076466/.18009457。
实际更新对alpha*g的最大余项约1.19e−7，属于FP32写回舍入，不作为新的精度问题。

| 每task四条（两state×两teacher） | P | RAW | RB |
| --- | ---: | ---: | ---: |
| 2 | 4 | 4 | 4 |
| 5 | 0 | 0 | 1 |
| 12 | 2 | 3 | 4 |
| 17 | 2 | 0 | 0 |
| 22 | 4 | 4 | 4 |
| 25 | 0 | 0 | 0 |
| 34 | 2 | 2 | 2 |
| 37 | 1 | 0 | 0 |
| 合计/32 | 15 | 13 | 15 |

RB−RAW保留13、新增2、丢0，+6.25pp，task/state联合95%描述区间[0,18.75]pp。
两条新增均来自teacher46：task5/state33和task12/state32；它们在**本批P中也失败**，不是恢复RAW丢掉的父成功。
teacher47的RAW/RB成功集合相同。RB−P保留12、新增3、丢3，净0，区间[−25,+25]pp；
RAW−P保留12、新增1、丢3，−6.25pp，[−25,+9.375]pp。
两候选共同丢掉task17/state33/teacher46与47，以及task37/state33/teacher46。
任务breadth P/RAW/RB为6/4/5；因此“与父同总分”不等于能力保持，不能藏起milk与Long任务的损失。

**预定判断。** H_noise_harm要求两teacher收益和父能力保持改善，本批没有兑现；只保留“一个teacher上新增两成功”的有限正观察。
方向变化经完整生成/策略路径传播到动作，不能说只是无功能作用的参数噪声；
但既没有恢复原三条损失，也没有超过父的净收益，不能将§148的计算机制升级为已验证的闭环修复。
小样本区间包含零，且没有独立训练/采集重复；不据此声称总体收益为零或修正必定无效。
本批关闭，不补原states34/35来追逐排名，不长训、不按成绩调alpha或重采梯度。

**功能与正常重放边界。** 32个首态×三对比的前五步夹爪sign都未改变，
RB−RAW的xyz RMS按task平均约.00418–.06028（环境命令单位，非米）；
task17的RB−P约.08093、RAW−P约.09999，虽差额减小但旧失败未恢复。
这只定位首态的真实控制变化，不能将幅度信用修正误读为“改变夹爪开合”，也不能用更接近P的动作证明整轨迹修复。
旧E2的同subset P为18/32，本批15/32，三条旧成功变失败单列，未拼入主对照；
该差异与两条新增同量级，提醒不能把单轮有限点值当可复现收益，不追逐bitwise一致或静默重跑。
三GPU阶段退出均0，完整计时相加.7020605942 GPU-hours；data0 1221.21MiB/代码504.16MiB在上限内。

**后继理由。** 旧梯度优化有探索J_Sigma，前后两批却只在另一teacher/state分布上检验J0。
仅扩大同类J0面板无法拆开这项事前保留的竞争解释。
[探索开关交叉](docs/designs/return_objective_alignment_design.md)固定P/已保存RB，不新增更新，
回到原八teacher×四初态，以一个新replica4作128条四格冻结诊断；主比较为更新在两目标上的差及其交互。
它只判断此有限更新是否在所优化分布上也无收益，或有探索收益却不迁移；
若无依据即不延长此回报更新分支，不把探索开关本身当部署修复，更不归为监督/视频问题的统一根因。

## 150. 回到原采集条件并打开探索也未兑现收益；关闭这一有限回报更新分支（2026-09-26）

Study `/data0/user/ymdai/ember_runs/return_objective_alignment_causality_20260925`。
16个bank及4条pilot来自E=29634cc9，余124条及分析来自E2=61974dee；design§6/59e7c1ad已登记唯一阶段例外。
E2只修共同seed前缀验收和精确来源绑定，policy/Writer/探索/rollout计算未改；未重物化或重跑E产物。
128唯一行、全T+1 trace/谓词、16full、所有worker退出0，缺项为空。

主讨论从每条原始completion、bank、trajectory和trace独立核对，重算五对比及20000次联合task/state bootstrap、
全部6382个replan的完整种子公式/加噪/实际动作、128首轮函数与GPU回执。
`coordination/main_recheck.{py,json}`保存审计；初态、注入、实际动作对trace的最大误差均0，
同模型J0/JS首轮加噪前均值完全相同。不是仅采信执行摘要。

| 原采集teacher，states0..3 | P_J0 | P_JS | RB_J0 | RB_JS |
| --- | ---: | ---: | ---: | ---: |
| task2 | 4 | 4 | 4 | 4 |
| task5 | 2 | 2 | 2 | 2 |
| task12 | 1 | 1 | 2 | 2 |
| task17 | 3 | 4 | 0 | 0 |
| task22 | 3 | 3 | 4 | 3 |
| task25 | 1 | 1 | 1 | 1 |
| task34 | 2 | 1 | 2 | 2 |
| task37 | 2 | 3 | 0 | 0 |
| 合计/32 | 18 | 19 | 15 | 14 |
| 有成功任务数/8 | 8 | 8 | 6 | 6 |

RB−P在J0为−3/32，R/G/L=13/2/5，95%描述区间[−.375,.125]；
在J_Sigma为−5/32，R/G/L=11/3/8，区间[−.5,.125]。
探索对P为+1/32、区间[−.09375,.1875]；对RB为−1/32、区间[−.125,0]；
交互Delta_Sigma−Delta0=−2/32，区间[−.21875,.09375]。
所有bootstrap共享重采样；没有把两个独立区间相减。每条件仅一个未参与梯度的探索replica4。

**预定判断。** H_objective要求Delta_Sigma>0、Delta0非正且交互为正，三个点值没有兑现这一模式；
没有证据支持“有效更新只藏在带探索执行中”。原teacher/初态上两种执行都没有净收益，
H_condition_transfer也没有获得支持；不能再把旧失败主要解释成只换了视频/初态或关闭噪声。
这不是等效性证明：区间宽，单次梯度/训练seed、单次探索重复仍无法排除较小或异质的真实效应。

milk task17与组合task37在RB两格均完全失去成功；不是只看净分的轻微交换。
§148已证实执行无关的夹爪幅度信用会实质改变完整Writer方向；§149和本批却没有验证它能修复这些损失。
因此计算机制成立，不等于旧下降原因已被充分解释。仅6/32组提供非零信用、两半方向近正交、有限更新非线性等
仍未互相分开；本批不证明所有RL、当前架构或视频学习无效。

裁决关闭§147–150这条有限回报方向/条件化/目标对齐分支，不加seed、步长、探索尺度或长RL来追逐正例。
下一项回到监督与真实生成函数的接口；新机制必须区别于已经做过的endpoint蒸馏和learner环境状态聚合，
不能把历史“专家监督有效过但不稳定”写成从未做过或完全无用。统一根因、可靠修复及有益视频必要增量仍未验证。
完整GPU进程回执合计.856028747小时（含物化/初始化），data0约1.2GiB、两代码树约504MiB，在原限内。

## 151. 通用训练/采样性质不足以解释EMBER差异；撤回去噪位置诊断的实施（2026-09-26）

Owner追问理论指导的出发点，并指出“若是VLA动作监督共有的问题，为何成为EMBER要解决的问题”。
主讨论重新审阅d7f500f0登记的flow_path_intervention，确认它没有证明当前差距主要受此机制限制，
也没有匹配的直接LoRA/语言生成参照来建立条件生成映射与该机制的交互。
第一步/第六步是人工预选的代表位置，不是原始证据已经定位的两个故障点。

需纠正的理论推断：训练使用条件插值样本、推理积分学习到的边缘速度场，本身不构成目标矛盾。
[Flow Matching原论文§3.1–3.2定理1/2](https://arxiv.org/html/2210.02747v2#S3)说明，在其概率路径和正则条件下，
条件目标与边缘目标只差参数无关常数，梯度相同；正确边缘场生成对应边缘概率路径。
单条条件直线不必等于某个噪声样本在边缘场中的积分轨迹。因此“两个位置不同”不能直接推出监督缺陷，
更不能推出所有VLA动作监督有问题。有限容量、拟合及积分误差仍可能存在，但须另外证明其作用范围。

旧设计的H0/H5干预最多检验何处借用另一完整expert的velocity有用；expert基础权重及训练offset不同，
纠正被后续迭代传播的长度也不同。即使取得收益，仍无法唯一识别训练覆盖，也不说明视频知识如何被编译、
为何语言参照较强或能力为何丢失。模型共有的风险只有在证明Writer放大它、或它限制本项目有益视频能力时，
才有充分理由作为EMBER主线。不能因一个接口尚未尝试过，就把“未覆盖”升级为“优先原因”。

裁决：撤回本批新增实现及GPU计算，保留已有代码/原件和原设计。17:18:33 UTC的Steer已送达实际Sol当前turn；
17:20:59 UTC核对Sol停止回应，并读取其`coordination/stop_point.json`：GPU及正式query/闭环均为零，
已有CPU实现保存在本地隔离分支，未集成。完整派发/停点来源另记progress。撤回不是实验non-pass，也不是全项目暂停。
后续理论必须解释匹配参照下的差异及多轮正负证据；当前没有替代GPU实验合同，不从历史active自动恢复。

## 152. 语言内容路径的函数包含与学习机制审计：候选构造不是根因证明（2026-09-26）

完整推导、源码owner、历史约束和CPU范围见
[语言内容路径审计](docs/analyses/language_content_path_audit_20260926.md)。没有新增模型forward、训练或闭环。
研究对象是匹配参照下视频生成没有稳定超过语言生成；不再由通用VLA训练/采样问题直接选择主线实验。
Owner要求每项架构/训练方案数学合理，并解释得通已存在的正负证据；一般恒等式与表达能力不是性能解释。

**已核对事实。** 现有B在Core语言blocks前使用text-only q，C使用FrameRead A(q,E)。
q参与attention Query但没有直接Value残差；E是含语言的逐帧多模态task-span与patch内容，不能称C没有语义。
FrameRead gate混合均匀/学习帧权重，不是回退到q的开关。这是精确结构差别，尚非失败原因。

**待验证构造。** 只加共享标量a，`u=A(q,E)+a*q`，a初值0。复制原C且a=0可保持C函数；
a=1、FrameRead output=0、Procedure modulation=0、共用模块取B参数时可得到B函数。
该构造包含现有两函数族，只给理想可达性，不给有限优化、泛化、能力保持或视频必要增量保证。
共享FactorHeads仍有跨层输出子空间约束，不能声称覆盖任意自由rank16 LoRA；旧private Writer胜freeAB反例继续保留。
新项给q一条直接内容/梯度路径，但有益梯度对齐未测，语言捷径也可能变强。

**历史限制。** 原架构存在correct/other较高、wrong较低的旧正例，否定“缺少q路径必然无法利用视频”。
四臂共同节点C−B先有+12/400、后有负值；不能预测全程统一代价。旧Compiler额外language query删除曾改善、
旧prior加residual未稳定保持，均反对“多加语言或残差必然有效”。新候选的Value位置与那些干预不同，仍须实证。
§144换W传播损害的范围包含多个模块，不是Core或本残差的单独证据；§145–150没有可靠修复，不叠加其学习改动。

**科学裁决。** H_path认为该参数化增加任务内容到LoRA的学习代价；竞争解释是视频知识未被有效识别/读取，
加q至多恢复语言能力，甚至强化语言捷径。候选只恢复B不算视频修复；没有超过C则不沿标量初值/尺度/层位扫描。
暂只交由Sol做真实owner模块上的合成中间表示CPU验证，核对构造、视频作用与梯度；它不证明任何行为能力。
GPU学习是否值得启动须另由主讨论形成匹配参照、有限节点与闭环裁决合同。当前没有经过验证的统一根因或修复。

补充完整节点审计：主讨论从四臂study原Source及B/C的held/seen results逐行复算6032行，
核对唯一task/state、共同env/policy seed前缀及B/C视频映射。630的B retained/gained/lost为48/76/10，
C为24/81/34；C−B=−19来自新增+5、额外损失24。Goal21差−32，其余七任务合计+13；
840/1050也呈C新增更多但损失更多，1260新增近同。seen64六节点C−B为+7/−10/−2/−3/−6/+1。
因此H_path须收紧到内容参数化下的保持/迁移代价，不能笼统声称整体学习较难；假如只提升新增能力，
却未减少原能力损失，不能确认这一解释。即使路径干预改善B基准，也仍要另证有益视频增量。

CPU构造41a0f4f4已完成：非零heads的76输出，a=0对C及指定构造对B maxabs均0；
视频扰动差.44040，padding余项6.41e−7，标量梯度及推导内积均.07022614。
Sol定向2项＋回归10项通过，主讨论独立复跑2项通过并审阅9行源码；没有真实source/video/GPU计算。
接续两臂C0/Cplus的fresh630及736条固定末点screen，保留B630冻结参照和同批Source；
这是直接检验上述架构预测的有界学习干预，不把CPU通过升级为原因确认。是否执行看progress及新design。

## 153. 内容路径的学习干预未改善保持；原C相对语言的缺口在小面板未复现（2026-09-26）

Study `/data0/user/ymdai/ember_runs/language_content_path_causality_20260926`，正式实现F=dca1b550。
C0/Cplus各fresh630、70560 queries；固定末点736条，全部来自同一冻结F，B bank只读43d801b1。
主讨论从原10份results、run contracts、736个连续trace及训练metrics/exposures独立核验，
重算42组配对差和20000次task-cluster bootstrap，核对43244个完整派生seed、28个固定full索引、
2520个事件/臂的实际video/query/noise/权重、6节点和worker退出；没有新增GPU计算或held标签。
证据：study下`coordination/main_recheck.py`及`main_recheck.json`；全部通过。

| 模型/条件 | held正确/80 | held另一正确/80 | seen正确/64 | held正确对Source的R/G/L |
| --- | ---: | ---: | ---: | --- |
| Source | 16 | — | 8 | — |
| B630语言 | 21 | — | 25 | 11/10/5 |
| C0原结构 | 24 | 30 | 27 | 11/13/5 |
| Cplus新增a*q | 16 | 19 | 27 | 6/10/10 |

Cplus−C0正确为−8/80，R/G/L=13/3/11，task-cluster描述性95%区间[-23.75,0]pp；
另一正确视频为−11/80，R/G/L=16/3/14，区间[-22.5,-5]pp。
C0在Source失败集合新增13，Cplus只保留8；Source成功损失5→10，主预测方向相反。
seen两者均27，但成功集合仍交换5/5，不能把总数相同称功能等效。
八task正确顺序0/1/14/15/20/21/36/38：C0=8/0/8/0/0/7/1/0，Cplus=6/0/7/0/0/2/0/1；
Goal21贡献−5，其余七任务合计−3；另一正确视频Goal差−2、其余七任务−9。不是仅一任务下降。

**干预实际作用及边界。** a最终−.00369846，629步归并后clip前梯度非零；C0/Cplus事件计划和实际训练查询一致。
`||a q||/||A||`有效token汇总的中位约.1997%，最大.5355%，末105步中位.2665%；新增路径数值作用较小。
这否定“路径未接入”，但不证明学习成了有益语言内容，更不说明负a代表“负语义”。
不能用非零梯度或分数下降宣布所有直接语言通路无效；本批检验的是该零初始化单标量、给定优化与有限窗口。
函数包含的CPU证明继续成立，却没有兑现有限学习的收益；不将它改写成优化保证。

**旧缺口不能当固定事实。** 新C0与B均丢Source成功5条，C0多新增3条；C0−B为+3/80，
区间[-6.25,13.75]pp，other−B为+9/80，区间[-1.25,23.75]pp。两者都不足以确认完整任务集合优势。
旧同80子集Source/B/C为16/24/22，新为16/21/24。Source旧新1得1失，B2得5失；
C0是重新训练，旧C→新C0为6得4失，不能把这种变化称同policy复评漂移。
旧C的Goal21为3/10，新为7/10；这正是旧全400主要负差所在任务，不能忽略。
本screen未复现额外保持损失，不由此推翻旧400事实；也不选旧/新C较差的一点制造可修复缺口。

**历史仍须共同解释。** 四臂六节点C−B先有+12后有负差，支持非统一劣势；
旧A900的Core/P交叉在固定donor Core下正确P使38→56/96，说明存在可消费的有益视频差异，
但不证明超过公平语言参照或真实时序必要。私有完整Writer14/20对共享3/20有可学习性正例；
当前更匹配private Writer16/32、freeAB14/32，以及保留原时钟的task-gradient隔离4/16对正常7/16，
阻止直接归罪输出容量或“删其它task梯度就能修复”。这些结果未被本轮阴性覆盖。
这些历史实验的Source、数据和任务范围不同，不拼成当前同一匹配矩阵，也不把旧私有拟合移植为当前视频增益。
原始来源分别在`runs/analysis/v52_core_procedure_cross_20260918/`、
`runs/analysis/pi05_ecp_prw_complete_single_task_20260906/`和
`/data0/user/ymdai/ember_runs/overnight_root_cause_20260922/{capacity_route_20260923,task_interference_probe}/`。

**裁决。** 关闭Cplus这项参数化的学习分支，不补它的400、节点或标量扫描；不称统一根因已经找到。
先按[固定C0完整覆盖合同](docs/designs/language_content_path_fixed400_design.md)补齐C0 correct/other、
B与Source尚未评的states10..49，共1280新行，无新训练/选点。它决定当前究竟存在什么相对参照的缺口，
不能单独证明视频因果或根因；既有80、后续320及合并400分开报告。小幅优势与宽区间不触发自动追加。
原批资源已计时15.35 GPU-hours，工程未端到端计时，总额未知；16.35含1小时规划预留，不能声称实测上界或已证预算合规。
科学矩阵无缺项，资源计量缺口如实保留；下一批明确记录所有GPU进程的完整时长。

## 154. 固定C0完整400没有复现旧相对缺口，视频有益增量仍未识别（2026-09-26）

Study `/data0/user/ymdai/ember_runs/language_content_path_fixed400_20260926`。
新GPU实现E=d18a8a0e；C0训练及原80行为F=dca1b550，B银行43d801b1，均按事前阶段合同保留。
主讨论直接读取8份原results/合同，核验1280新增唯一行、全部新增T+1 trace/谓词、16固定full索引、
22182个共同replan种子前缀、640新C银行来源及每task两轮50视频覆盖，复算新增/合并全部24个有向配对与共同bootstrap。
`coordination/main_recheck.{py,json}`通过；首次只读检查因错误假定信息墙字段名退出，已按原schema修正，未改变任何GPU原件。

| 模型/条件 | 新增320 | 合并400 | 合并对Source的R/G/L |
| --- | ---: | ---: | --- |
| Source | 41 | 57 | — |
| 语言B630 | 100 | 121 | 47/74/10 |
| C0正确视频 | 96 | 120 | 43/77/14 |
| C0另一正确视频 | 102 | 132 | 46/86/11 |

合并C0 correct−B=-1/400，R/G/L=93/27/28，task-cluster描述性95%区间[-3.5,+2.75]pp；
other−B=+11/400，100/32/21，[-.25,+6]pp。新增320两项为−4/+2，区间[-6.5625,+3.4375]/[-2.5,+4.375]pp。
other−correct合并+12/400，106/26/14，区间[+.75,+5.5]pp；新增320为+6，区间[-.3125,+4.6875]pp。
两轮都使用各task同一50视频池各一次，差额不来自挑视频池；有限state-video配对、非线性闭环及正常复评变化仍在。
不按other高点选择模型，也不把总分/区间差异解释成某条视频的质量排名。

任务顺序0/1/14/15/20/21/36/38：Source=17/0/0/0/0/38/1/1；B=32/0/44/0/1/41/0/3；
C0 correct=34/0/40/3/0/38/1/4；other=36/0/45/6/0/40/1/4。breadth为4/5/6/6，Long仍弱。
Goal21的C0−B仅−3，其他七任务+2；旧C630−B630的−32/+13不在本次新训练C0中复现。
不能将新C0变化归因于某项架构修复：C0架构保持原样，训练物理拓扑/实际数值轨迹与旧C不同，非单因素成因实验。
C0相对Source有63/75净增，但语言B也有64净增；这不建立视频贡献，亦不证明两模型等效。
原80看过后才登记补320，单训练seed、8个诊断task；合并400不是新的全盲测试、官方方法资格或相邻稳定性证据。

**特征层解释边界。** 相同Core/Procedure计算图已经能在本次达到接近语言的整体能力，
故不能继续把“语言Value未直连”当必然性能损害；也不能从排名反推E/P已含有可迁移操作知识。
E含语言与逐帧视觉内容，H是native动作计算响应，当前成绩仍兼容初始场景/语言足够、变化特征未被有益读取、
以及已有局部视频贡献被其它损失抵消。对已冻结checkpoint的CPU参数读取显示C0各head帧汇聚lambda=.04994–.05062，
Cplus=.04955–.05046；这是均匀/选择混合比例，不是信息损失率或已验证根因。
读取记录在原language-content study的`coordination/main_frame_gate_check.json`，无模型前向或新GPU计算。
后继[原生特征变化干预](docs/designs/native_feature_change_causality_design.md)固定q与完整Writer，
交叉保留/首帧广播E、H，同步重算H初始化路径，并记录特征→LoRA→首轮动作→闭环。
00仍保留初始内容和原时钟，不是learned static baseline；混合格有联合分布偏移，正差不能自动称视频必要性或修复。
这是新的有界320行诊断，不因本400差异不明确而自动补seed/节点或开启新训练。

实际完整新GPU账5.904167小时（物化.341944，四评测1.363333/1.413333/1.422222/1.363333），低于8；
study约4.535GiB、开发+formal约492.97MiB。验收脚本别名/region字段修正只影响CPU读取，E与全部有效行未重跑。

## 155. 逐帧原生特征在冻结C0中有有益作用；其额外学习价值仍待匹配参照（2026-09-26）

Study `/data0/user/ymdai/ember_runs/native_feature_change_causality_20260926`，统一GPU实现eb9a373d，
C0@630训练来源dca1b550。主讨论审阅原生投影/Compiler/物化及评测diff，从8份原results/合同、320条
trace/trajectory和80×4份特征独立核验；943个full replan双相机、4491个共同seed前缀、全部首态与实际动作一致。
成功即停时只核对已执行T步，不能要求最后保存的五步计划全部被执行。全部worker exit0，六差及共同bootstrap复算通过。
证据为study `coordination/main_recheck.{py,json}`，不新增GPU/环境/teacher标签。

| 特征内容 | 成功/80 | 各task成功，顺序0/1/14/15/20/21/36/38 |
| --- | ---: | --- |
| E0/H0：首帧内容＋原时钟 | 20 | 7/0/5/0/0/6/1/1 |
| E1/H0：逐帧E、首帧H | 24 | 8/0/8/1/0/6/0/1 |
| E0/H1：首帧E、逐帧H | 20 | 8/0/5/1/0/5/1/0 |
| E1/H1：完整原生内容 | 27 | 9/0/8/0/0/7/1/2 |

主要11−00为+7/80，R/G/L=18/9/2，task-cluster描述性95%区间[2.5,16.25]pp；
task净差+2/0/+3/0/0/+1/0/+1，净增分布四task，breadth四格均5。
其它预定差：10−00 +4，15/9/5，区间[-1.28125,13.75]pp；11−01 +7，19/8/1，[0,17.5]pp；
01−00 0，16/4/4，[-5,5]pp；11−10 +3，20/7/4，[-1.25,8.75]pp；交互+3，[-5,11.25]pp。
不能把H单独净差0解释为没有被使用，也没有足够证据宣布E/H协同机制已确认。
旧11为24/80，本次27，3得0失单列；旧Source16成功中本次四格保留8/10/9/12，仅历史分层，不作新Source配对收益。

**到特征层的已证和未证。** 四格q/位置/mask相同，E0/H0逐视频广播实际首帧，H0同步重建I且保留50个位置。
固定E只改变H时，Core严格相同，变化经Procedure/P及调制进入完整LoRA；E同时进入Core与Procedure，
因此E简单效应不能归到Core单独一个模块。保存的feature/LoRA/首轮函数均有变化，但范数差不说明语义正确。
11−00中首帧以外内容在当前冻结模型/80条件上有净执行价值；它可能包含多状态视觉覆盖、终态线索及操作关系，
不是已经分离出的运动/顺序知识。混合E/H仍有联合分布偏移，00也不是充分学习的static/no-video参照。

FrameRead在实数计算中是均值Value加选择性修正。保存BF16 attention的sum并非严格1，
`sum(w-1/T)V=(sum w-1)mean(V)+sum w(V-mean(V))`必须区别权重和误差与内容选择。
CPU从原权重/特征复算：完整输入同读出空间的均值RMS1.330489，中心内容选择.00146468，权重和项.00225337；
首帧广播的中心内容选择仅9.33e-8，原分析约.00306的“selective correction”实际来自权重和项。
恒等式最大余项9.69e-6；这不是需要修dtype的bug，也不能由约.11%的选择项宣布其功能无用。
Core入口以多帧内容汇聚为主的数值事实，加上E还走Procedure，要求先辨识内容价值，不能直接调gate或删H。
读数保存在`coordination/frame_read_content_accounting.json`，是post-hoc数值拆分，不是新语义因果证据。

**原始分析单位修正。** `paired_comparisons.applied_osc_first5`实际取自环境action，并未controller clip/scale。
主讨论核对安装的OSC_POSE及LIBERO默认调用，用原保存动作CPU重算；scaled OSC xy的11−00 RMS为.00351927，
不是原字段的.07038591；两者均不是实际EEF位移。原文件保留、更正在`coordination/action_units_amendment.json`，
所有闭环、输入配对和成功统计不受影响。后续统一明确normalized/environment/scaled-OSC三个单位，不追查正常低位差。

**科学裁决。** “当前C0完全没有有益地使用首帧之外内容”的强说法被这批有限证据削弱；
“已经学到超过语言的教学能力”仍没有建立，前批400的120/132对121不被这次损伤差替代。
不由冻结27对20直接加新模块，也不补这四格400。登记一个fresh的首帧内容学习参照：相同C0初始化/fit28/FM/
70560查询/630更新，只有E/H内容投影改变；原C0冻结复用，六面板448条新配对闭环。
数学变量、sum梯度、已有静态对照的真实范围及可改变判断的结果分支见
[学习首帧参照合同](docs/designs/learned_initial_content_causality_design.md)。它检验学习后额外内容价值，
不承诺一轮定位所有根因；仍不称充分收敛、部署资格或统一修复。

完整GPU账1.561389小时，data0约4.594GiB、代码505.56MiB，均在限内。
三格remaining实际在gpu02，本地外层回执误写gpu01；主讨论以worker UUID、evaluator live准入和双节点快照核对实际位置。
本批评测峰值4卡，快照没有另一个ymdai任务占用；E1/H1两次错误节点尝试在worker启动前被拒，零有效行，随后gpu01续跑。
保留`launch/node_mismatch_amendment.json`及失败原件，不重跑有效数据；下批启动前绑定真实host/node/UUID。

## 156. 长期推进错误的回顾与跨会话决策约束（2026-09-26）

Owner要求回顾过去几个月，并明确要求持久记录，避免本轮结束或新会话后遗忘。本节是研究决策回顾，
不是新科学实验结果或统一根因。已核对7月底以来的研究索引、相关旧评审与数学构造、46组审计，
以及匹配曝光、固定reader、纠正oracle等代表性原件；不声称本轮重算了全部历史闭环。
稳定要求见[Owner要求](docs/current_owner_requirements.md)，当前方法工作与唯一实验合同见task_plan/progress。

### 实际反复发生的错误及其证据

| 错误 | 具体记录/反例 | 今后应改变的判断 |
| --- | --- | --- |
| 局部问题替代EMBER目标 | 8月底G3/primal/PNBTT的局部接口迭代；近期从早期动作追到回报信用、探索、flow位置，后者缺少条件生成链的差异化依据而撤回（§151，零GPU运行） | 解释视频到LoRA的实际不足；通用VLA性质须先建立与EMBER的具体交互，不能因“未证实所有模型都有”就称特有问题 |
| 可表示、能学习、能迁移被混同 | 共同LoRA下界曾反复需要澄清；自由参数/任务专属/privileged正控不等于合法RGB共享获取；Cplus函数包含未兑现保持预测（§152–153） | 下界不是公共底座课程、优化保证或rank免责；正控只裁决实际接口，架构/训练仍需解释学习机制 |
| 局部公式或几何被当成性能原因 | 旧评审把GOMQ151→136解释为rank32→16容量损失，实际A32=[A0;A0]，BA=(B0+deltaB)A0，有效rank≤16；低梯度retention也不单独证明冲突 | 核对实际计算、假设和功能作用，不以范数/夹角/数学复杂度命名根因 |
| 视频敏感性替代有益教学 | 历史内部correct/wrong容量—间隔冲突；zero-LoRA/no-video不是learned language参照；当前正确120/另一正确132/语言121也尚未证明稳定增量 | 先看正确输入绝对收益与保持；wrong下降、参数差异、顺序扰动掉分不能单独证明教学价值，同task相近LoRA也可能是正确抽象 |
| “原生/完整/有时序”被当成语义保证 | G1的scalar Y-span投影120→109/250，Goal11→0、Long8→0；完整H、有效梯度与标准模块也曾未兑现共享闭环 | 区分读取原生知识和强制输出在其span；H不是teacher动作，轴/槽位不自动等于物体/阶段，具体模块不是必须保留的特色 |
| 证据未累计约束主假设 | §62已明确指出：每个负结果都限定外推边界，却未相应下调主假设支持度和投入；旧PNBTT E1曾被过度解释成未运行的真实Program E2失败；早期v5.2/v6正例必须保留 | 同时更新“排除了什么”和“现在还值得投入什么”；不能由尚未普遍证伪无限续修，也不能由受限阴性否定整条路线 |
| 对照与数据/预算概念混淆 | 同每task150次视频访问，v5.2 old/TC132/51，v6为95/111，但Adam次数/LR等同时变；视频池/K/任务映射数不同；旧400行只有255个task-video条件的正式身份已撤销 | 明确独立任务、条件、query、更新和学习时钟；总分差只裁决匹配变量；未执行、工程失败、短预算、科学阴性分别记录 |
| 参数稳定或净分掩盖能力交换 | LPCP近.999995的BA cosine仍有42个success-set交换；139→143包含23得19失；单点能力、另一点特异性不能拼成虚构方法 | 查看同状态R/G/L、任务覆盖与实际保持，不能把平均数相近叫学会同一能力 |
| 评测/工程形式支配科学投入 | 关系支持由21868→9932缩为1500 stage1；被动采集/CPU验收窄修曾受单提交形式阻塞；执行批次停止不等于主讨论可以等Owner催促 | 先取得能改变判断的最小证据，再按缺口补；保护计算和原件而非为形式重跑；主讨论持续推理且不为让GPU不停而派无依据实验 |
| 沟通把不确定解释说成技术对象 | “解决MT-BC的失败”实际上只是指其未完成某评测回合；工程完成/实验完成/根因修复也曾没有清楚区分 | 直接说明事实、假设、未知和下一决策；MT-BC仅为最终性能参照，不是待修复的研究对象 |

相关完整入口：[7–9月主线](docs/research_history.md#研究主线速览)、
[46组审计](docs/analyses/v52_evidence_audit_20260917.md)、本文件§62、§151–155。
本轮复核的原件包括：
`runs/outputs/pi05_as_writer_v52_v6_recipe_matched_exposure_seed7_20260801/analysis.json`；
`runs/analysis/video_functional_20260911/native_reader_analysis.json`（末点24/24任务仍弱于LoRA学生）；
`runs/analysis/native_correction_writer_20260913/oracle_rollout/decision.json`（90/384对复用Source68/384，明确privileged）；
Git `ac233fa0:docs/gomq_rank16_archival_card_20260824.md`及
`fcdb6e43:docs/expert_review_20260902_global_route_reassessment.md`中的原错误归因与后续审计纠正。
这些原件的正负结果有各自合法范围，不合并为一个已验证的统一机制。

### 首帧讨论后的明确结论

Owner认可学习首帧对照的出发点：区分冻结模型对输入的依赖与同预算学习后的额外收益。
此前主讨论承认其尚不能直接给出架构修复，不应进一步把它整体贬为没有价值。
合理诊断无需每次涨分或单独给出最终架构；问题是连续诊断是否实质约束了方法选择。

- S0与C0总分接近，先分相同成功集合与收益/损失抵消；不宣布视频无信息或已有根因。
- 重新学习若恢复冻结删帧丢失的同一批能力，削弱其不可由当前首帧条件学到的解释；
  不证明无限预算能力上限。S0还保留原时钟/长度，不称纯单图。
- full胜出支持有限设置中的后续内容收益，不区分视觉覆盖、状态变化和操作顺序，也不自动决定加强动态模块。
- 当前C0低于或近似语言的事实不能未经核对从旧节点泛化；B语言Writer也不是MT-BC。
- 目标仍是提高EMBER绝对能力、建立正确教学的有益贡献并保持能力，最后匹配比较强MT-BC；
  不必先诊断MT-BC，不围绕所有删减输入差额无限追加。

### 对后续工作实际生效的约束

每次主线投入前，主讨论须完成：具体待解释事实→竞争机制→特征/算子干预→不同结果如何改变方法判断。
方法提案须说明从q/E/H到过程表示、完整LoRA和执行状态条件作用的全链计算，以及训练为何学习这条联系。
从历史反例约束一个首选机制，必要时用架构/训练改动本身验证；不能要求先穷尽所有诊断或先确诊统一根因。
阴性要降低对应机制的支持度和后续优先级；仅有代理改善不算修复，涨分但机制未识别只算经验收益。
上述原始记录形成于S0执行期间；S0现已完成并独立验收，见§159。当前执行者及Reader工程边界以progress为准，
不能从本节历史叙述恢复S0或旧Sol任务。

### 接管后补记：连续换架构与分模块过关的两类循环

Owner再次要求回顾过去几个月，特别是“理论上提出好几版架构但性能都不行”与“逐模块过关却陷入局部问题”，
并追问何时会推翻v5.2继承基础、如何避免换架构本身也成为循环。以下区分复核事实与新主讨论的决策承诺；
Owner没有断言v5.2路线已被证明不可能，主讨论也没有已经验证的替代架构。

**代表性历史链及其实际边界：**

| 时段与推进链 | 复核到的事实 | 应吸取的教训 |
| --- | --- | --- |
| 7月底至8月：v5.2、v6、Target-Owned及Dynamic-K后继 | 同每task150次视频访问，v5.2 old/TC为132/51，v6为95/111；更新次数、LR时钟等并未匹配。Target-Owned放开共享并增加容量后为99→76→86→68（均为各自原validation400）；Dynamic-K还同时改变读取、出口和学习条件。v5.2后续也真实跑到1800，不能只保留900的132；v6及GOMQ的有限正例不能抹去 | 一个结构听起来更合理，不足以保证在实际学习条件下更强。每轮改多个部件后，再用下一套理论解释下降，既不能定位原因，也不能累计形成可靠方法；不能把不同合同总分排成单变量架构排名 |
| 8月底至9月初：G1/G2/G3、primal、bank interaction、PNBTT | G1的保留正例依赖privileged求解，G2只证明动态读出；J2 task-local正控10/10通过，真实Program共享获取仍弱。EBSRI S2是固定task-token路线且整体未过，Natural Program S3未启动；PNBTT只运行free-query E1，E2未运行 | 可表示、局部可解、自然视频共享可学、闭环迁移是不同命题。局部通过时使用的输入、自由度和优化条件不能隐去；多个局部正例不能拼成尚未执行的端到端成功。未运行后段也不能反过来登记为整条路线失败 |
| 9月11–12日：Video Functional、额外reader、去蒸馏、VL适配、去辅助 | 首版及匹配frame_set没有可信有序收益；两个额外读出的离线功能拟合仍逐task弱于LoRA学生。后继保留了部分训练获取，却未兑现迁移与保持。§62当时已指出“局部干预结束、主假设几乎不变”，§65才明确关闭该设计及局部修补投入 | 每次只写本轮non-pass并保留所有上层解释，会让后续实验不断获得同一个理由。必须合并证据下调“补执行信用/解除某个冻结边界即可得到可迁移教学”的具体充分性预期，不靠尚未普遍证伪继续保护它 |
| 9月13–14日：真实纠正oracle、Native Correction、Semantic Path、Local Field、Process Pullback | privileged纠正90/384对复用Source68/384有有限闭环收益；合法共享模型未继承该资格。Semantic Path有序validation77→48，Local Field48→47（Source47）；Pullback在300/600/900更新为64/72/64（以上共享模型均为各自validation400），有局部训练获取但无广泛稳定迁移。各版同时变化的读取、出口、监督及预算均保留 | oracle存在性、rank16/PCA出口的功能前提不能替下一套合法读取与学习背书。若共享获取预测反复未兑现，下一步不能自动再换坐标、局部头或监督；这些结果也不能被合并成所有RGB获取不可能 |
| 9月25–26日：动作/回报信用、探索对齐、拟议flow位置诊断 | §147–150有真实有限作用证据，但删除无关夹爪信用、恢复原采集条件或打开探索没有兑现预期能力收益；flow位置诊断因缺少与EMBER差异的联系撤回，GPU运行为零（§151） | 解释了一个局部计算效应，不等于解释了视频到LoRA的能力缺口。必须允许在投入前撤回缺乏判别价值的诊断；撤回不是实验阴性，撤回一个批次也不是停止方法研究 |

本次沿[46组历史审计](docs/analyses/v52_evidence_audit_20260917.md)及
[共享原生编译原论证§12](docs/analyses/temporal_control_compilation_theory.md#12-共享原生编译学习的历史证据口径2026-09-14)
读取相关完整边界，并直接复核以下现存原始汇总/裁决：上述v5.2/v6匹配曝光`analysis.json`；
`runs/outputs/pi05_ecp_j2_functional_positive_control_10task_c4704cb_gpu01p012345_20260829/aggregate.json`；
`runs/outputs/pi05_ecp_event_bank_set_s2_direct_functional_gate_25477c9_eval3e15632_gpu01p013456_w6_20260901/aggregate.json`；
`runs/outputs/pi05_ecp_pnbtt_e1_gate_aligned_necessity_s110_e65c6388_gpu01p12_20260902/evaluations/qualification.json`；
`runs/analysis/{semantic_path_writer,local_correction_field_writer}_20260914/bounded100_decision.json`及
`runs/analysis/process_pullback_writer_20260914/bounded900_decision.json`。§62–65、§91–100与§150–151的既有论证保留。
本次没有重算全部历史raw rows、加载旧模型或重跑实验；历史sealed controls不转为新设计/选点数据。

**两类循环的共同决策错误。** 单次实验有停止线还不够：若负结果只能结束当前名称，却不能降低共同假设的优先级，
下一版仍会以同一理由重新获得预算。模块过关同样不能形成自动晋级链：局部条件改变后，原来通过的命题未必适用于组合系统。
这不否定合理诊断、分阶段控成本或整体重构；错误在于让它们替代完整方法的可失败预测与实际能力裁决。

**新主讨论对刚才回复的持续承诺：**

1. **目标固定，架构基础可撤换。** 保留合法输入、信息墙、一次完整LoRA及绝对能力/视频收益/保持目标，
   不保护v5.2血统、Core/P、slot、FactorHeads或当前Reader。早期正例约束解释，不给继承架构永久优先权。
   当前证据已足以开展基础假设审查；这是研究优先级判断，不是整条v5.2路线已被证伪。
2. **在主假设层累计证据与投入。** 每项实质后继注明继承哪个学习假设、与最近等价历史究竟改变什么；
   改名、换模块或换session不清零负证据及已付成本。相关失败不算独立投票，短预算、合同变化和正例都必须计入；
   同时不能以这些限制为理由，永远不下调投入。每轮核对预期/观察、支持度变化、关闭的分支及下一投入理由。
3. **基础重构的依据要能改变方法选择。** 以下情形可支持撤换相应基础，而不要求先穷尽诊断或证明普遍不可能：
   已确认的结构约束阻断目标所需信息/函数/学习路径，且剩余输入与算子不能补足；有判别力的联合证据持续削弱共同学习前提；
   或已有完整替代原则及证据理由，使其预期信息价值和目标收益优于继续局部修补。
   仅有低分、小梯度、低rank或某个消失方向，不满足结构根因证明；尚无可靠替代时如实保留未知，不为了换代而拼新图。
4. **重构先给完整学习与执行原理。** 在选择模块之前交代合法特征来源、信息如何组合、生成哪些完整参数、
   标签与梯度如何训练该映射、参数怎样作用于自身执行状态，并说明关键竞争预测及传递风险。
   数学自洽和可表示性不是可学性担保。允许一次有原则的整体改版，不强制拆成逐模块Gate；整版比较只裁决整版，不能唯一归因某模块。
5. **参照不随失败下移。** 用同协议下有效的强参照判断进步，同时保留匹配的直接对照；不能只击败上一版失败模型。
   新架构明显变差后不自动成为下一轮底座；保留可靠参照的证据与必要可用资产，重新评估共同原则。
   不同source/split/面板的历史峰值不能充当匹配参照，旧小幅正例或不稳定峰值也不冒称强基线。
6. **区分工程纠错与科学换题。** 已证实的合同错误按其影响修复；正确执行后的科学阴性不能改称bug再自动补丁。
   每批预先给wall-clock预期、GPU/产物预算与会改变选择的停止分支，依据实测逐步投入；
   不把小screen投资线当统计资格，不规定每次必须涨分，也不因“明早交进度”启动大矩阵。
7. **这些约束立即约束Reader自身。** 当前只是有界工程候选，BAh已有query信用，S0局部初始化问题也不解释C0近似语言。
   工程通过不证明有益教学；若后续Reader学习不获益，应降低/关闭本候选，不能自动加层、rank、seed或转蒸馏。
   只胜Writer但不胜匹配语言读出，仍没有视频教师；即使教师有益，传给唯一完整LoRA也须由学生实际闭环证明。
   不把“Reader→语言参照→编译”变成新一套不可撤销的模块课程；若传递只改善局部loss，不能称完成EMBER修复。

上述承诺用于后续自主科学裁决，不增设Owner审批、强制诊断清单或新的GPU授权；现有Reader工程继续按原范围执行。

### Owner最后提醒：避免近似重犯，并立即检查Reader自身

Owner明确要求不要提出与几个月以来失败架构/训练几乎一致的改进。比较单位必须是实际特征、算子、标签与梯度、
学习条件和部署作用，不是模块名或版本号；近似旧图须有新证据或能改变旧失败预测的实质机制，不能只说组合尚未试过。
须读相关完整旧方案及后继修订，不能挑其中较弱的固定表示探针来证明当前方案“不同”。不由此重跑旧实验。

这一要求已实际改变当前候选判断：[机制分析§6.1](docs/analyses/feature_to_operator_mechanism_20260926.md#61-最近似历史的完整比较与本次降级)
核对Video Functional§2–4/§7–11、`a8964874`的memory cotangent/encoder重放及`b1d3fa25`的真实native hook。
旧路线已有联合encoder/执行reader学习及VL Meta版本；联合学习、真实FM、query依赖和零head不是新机制。
当前C/P、独立q/v纠正及单reader目标有实际计算差异，但未证实它们对应旧失败原因；此前只突出固定E诊断不足以支持优先重访。
主讨论因此下调Reader正式学习优先级，撤回以“这次联合训练”作为新投入理由。原工程按界限收尾，630准备稿保持未激活；
工程通过、梯度接通或成本低均不替代新的科学理由。尚未证明Reader普遍不可能，也不为避免重复而立即拼另一版架构。

## 157. 特征到执行作用的计算审计与下一候选（2026-09-26）

本节是代码/理论及方法选择记录，不是新的科学实验结果。完整论证见
[特征到执行算子](docs/analyses/feature_to_operator_mechanism_20260926.md)，具体准备边界见
[原生条件读出准备稿](docs/designs/native_conditional_reader_design.md)；尚未派发，不替代active S0。

- 当前E同时含图文内容，H是固定公开noise/tau1的原生响应，不是teacher正确动作；P真实重读H及前后E。
  Core帧汇聚与P中心化有明确计算性质，但不能由此直接宣称丢失了实际有用操作知识。
  完整LoRA通过`BA h(current state)`本来就能状态条件执行，不把单层线性误写成整个策略不能选择阶段。
- 主FM的完整LoRA cotangent累积与同版本Writer VJP是正确链式信用，跨episode监督本来存在。
  其间接条件映射可能影响获取，但公式和梯度路径不自动定位优化根因。
- 旧Semantic-Path、统一原生图和Video Functional反对“保留轴/进入native/加辅助即可充分”；
  固定native reader的24/24任务仍弱于学生尤其反对立即复活弱教师蒸馏。
- 因此优先形成一个可失败候选：fresh联合教学读取与执行query的原生功能读出，先测它是否真正获得值得传递的
  正确视频闭环能力，再检验同坐标作用能否由完整LoRA承接。拟定作用点是原生第10层q/v两个真实target，
  不把1024维hidden纠正与2048/256维投影纠正混为同一标签。它是诊断/教师，不能计入EMBER部署成绩。
- 用attention Key/Value导数说明query如何给教学内容信用；固定查询rank约束回归可拆分局部线性残差与秩代价。
  合成CPU核对两式分别在5.33e-12及2.84e-14误差内成立，只验证数学，不是模型证据。
  查询数不足/列满秩、跨video混池及深层h变化均限制局部回归解释，不因公式漂亮先扩hidden探针或rank。
- coverage强MT-BC selected300=155/400与同协议Writer600=142/400保留；其训练用过当前diagnostic-held任务。
  当前C0的120/400与155不直接相减；A为匹配direct16，B为language Writer，均不偷换成充分训练的强MT-BC。

准备稿采用先R_V、取得实质绝对能力信号后再投入R_L的分阶段方式。Reader优于C0却不优于language读出，
仍不能称视频知识教师；Reader无益则降低该候选支持度，不自动蒸馏或换层续试。
只有最终唯一LoRA的绝对能力、有益教学及保持成立，才进入完整配对与强MT-BC验证。

## 158. LoRA已有query级信用；非线性读出的收益不能自动编译（2026-09-26）

这是在S0执行期间独立完成的数学审查，不是新模型实验。完整推导在
[特征到算子§10](docs/analyses/feature_to_operator_mechanism_20260926.md#10-独立推进的传递审查信用等价计算差别与下一裁决)。

对`delta_y=sum b_i(a_i^T h)`，`dL/db_i=g(a_i^T h)`、`dL/da_i=h(b_i^T g)`；
上游教学表示已通过这两侧获得执行query级信用。把同一K/V线性读取写成`V K^T h`再一次性编译成BA，
在相同参数化/训练图下连梯度也相同。因此“换成reader就得到缺失的状态信用”不能成为方法依据。

候选真正不同的是执行h出现后，重新softmax读取完整C/P并进行query相关非线性调制；
这可能改变有限学习偏置，却增加了最终编译困难。tanh标量例证明单target固定线性作用不一般等价于softmax读出；
不将此反例扩展为全38-target深层LoRA不可表达。局部误差对后续动作的作用还依赖真实Jacobian，不由L2单独决定。

合成FP64 CPU的前向最大差1.78e-14、各输入梯度最大差7.28e-12，A/B梯度公式差0；
它们只检验等价式，不是模型梯度诊断。没有读取teacher标签、执行新policy/环境或新GPU。
源码与小输出在`.codex/tmp/native_reader_transfer_review_20260926/`；推导在本仓完整保存。

原生Reader候选保留，但收紧为“联合非线性内容选择能否学到值得传递的有益教学控制函数”。
现有Writer适配38处、候选只适配2处，函数类没有已证包含；不能由单次胜负唯一定位Compiler或表示。
正结果才支持后继编译研究，尚未选定局部L2/KD；负结果降低该具体路线优先级，不以增加层/头反复救活。
旧空间投影、private Writer/freeAB及head缩步反例继续约束选择，不因读出措辞重新认定输出头是根因。
S0仍唯一active，主讨论的持续分析及验收准备不以其结束为前提。
另纠正本轮准备稿的一项实际输入错误：C0教学侧是agentview单相机/256patch，执行侧及full-case采集才是双相机。
以原F的C0 run_contract、配置和encoder定义核定，准备稿保持这一输入；未曾派发或按误写的512patch/双相机启动。
在未读S0整批成绩和任何Reader成绩时，进一步收紧候选投资规则：Reader须相对C0/S0中较高correct净增至少8/80，
other不低于两个参照的较高值、correct至少两suite净增，才默认投入语言Reader。只按整面板总数取参照，
不按state拼接oracle；这是成本决策线，不是正式选点/显著性或方法资格。不以超过较弱C0忽略已有首帧能力。

## 159. 首帧学习未追平；过程路径的初始化交互限制信息归因（2026-09-26）

S0 study `learned_initial_content_causality_20260926`完整结束。主讨论独立核对原results、bank、448条唯一行、
全部T+1/动作/谓词/初态及seed前缀、24双相机full、8阶段worker退出；重算全部14有向比较和20000次task-bootstrap。
另核定70560查询、2520事件、28task各90访问、两臂事件plan/实际teacher与跨episode查询/LR完全匹配。
证据在study `coordination/main_recheck.{py,json}`、`main_training_recheck.{py,json}`。

| 面板 | C0完整内容 | S0首帧学习 | C0相对S0 R/G/L | 成功率差及描述性95%区间 |
| --- | ---: | ---: | --- | --- |
| held correct80 | 27 | 22 | 18/9/4 | +6.25pp，[-3.75,17.5]pp |
| held other80 | 27 | 17 | 14/13/3 | +12.5pp，[1.25,30]pp |
| seen64 | 26 | 22 | 13/13/9 | +6.25pp，[-10.9375,23.4375]pp |

held按task0/1/14/15/20/21/36/38，correct的C0−S0为0/0/+4/−1/0/−1/+1/+2；
other为0/0/+7/+1/0/+1/+1/0。correct总breadth5对4，seen两者11/16，但有22个成功集合交换。
因此不是统一上移或已掌握同一能力。主区间较宽、单seed/固定630不证明等效、信息上限或架构资格。
C0旧三面板24/30/27到本批27/27/26的历史self变化另存`main_historical_self.json`，未借旧行抬高或压低比较。

### 新识别的特征/梯度事实

在未读本批完整分数前，已退出训练检查发现S0 Action Meta仅450..630共181步非零，C0为3..630共628步。
这是实际参数梯度在跨rank归并后、clip前的记录。S0的105/210/315/420四个checkpoint中，
Compiler modulation、两层H读出、两层E读出及72个Action Meta B张量均仍为零；525/630已活动。
定向检查不涉及全树hash或追求低位一致，原件`main_static_path_audit.{py,json}`。

机制分析§11给出具体解释边界：fresh S0的H/E读取输出初始为0，重复I经Q/K-only RoPE与逐位置模块保持相同P；
中心化使Procedure Value为0，后接零初始化且无bias的modulation矩阵W。
于是`dL/dW = g RMS(r)^T = 0`且`dL/dr = J_RMS^T W^T g = 0`，精确算术下形成不动子空间。
实际Compiler合成CPU检验复现双零；变化P先启动W，再一次W更新启动P梯度，Core始终可学。
这说明首帧干预同时改变了内容和这个架构的有效学习路径；不是违反原合同的工程错误。
本批真实数值轨迹450步后打破零状态的具体事件尚未识别，不追加精度、kernel或逐bit调查。

### 科学裁决与后继投入

“现有后续内容收益能被同预算首帧重学全部恢复”受到削弱，但不能将C0胜出唯一归因于操作知识或顺序。
旧冻结后续内容正效应不被抹去；完整C0从第3步即可学习Action Meta，所以这里也没有解释它与语言近似持平。
本批结束，不重训/修首帧、补400、延训或增设删减对照。当前仍没有统一根因或验证修复。

保留§157–158的联合执行条件读出作为有限候选，只激活其工程准备：验证q/E/H→C/P→实际执行q/v残差的
真实梯度与canonical闭环接口及成本。其非线性内容选择是待测假设，不称为补充原LoRA缺失信用。
工程上限0.75完整GPUh、2GiB新data0、6条train-only接口episode；正式630/held/语言第二臂/蒸馏均未授权接续。
完成后由主讨论核验，才确定是否值得投入学习；它仍是诊断教师而非EMBER最终单LoRA部署。

S0正式GPU来源4f6c75e4，两个held-other使用仅CPU来源校验修正b9e15410；原F的224 C0银行只读，
新S0银行224，科学模型与原件未重跑。完整计量4.669167 GPUh≤9，study约4.48GiB≤10；没有登记科学原件缺项。

## 160. Reader工程成立，但不足以支持重访旧学习假设；正式学习启动前撤回（2026-09-26）

本项完成的是工程合同，不是视频能力实验。执行者最终推送`2fbd4c7d`；主讨论核对最终四个新增文件及两项窄修，
独立运行5个CPU检查并逐项读取已结束的原始metrics、checkpoint trainer/rank状态、六条JSON/npz/PT。
研究根`/data1/user/ymdai/ember_runs/native_conditional_reader_engineering_20260926`；
`engineering/main_acceptance.{py,json}`保存主讨论的独立核验，未运行新模型/环境、未重跑有效训练。

- 两模式各fresh4及2→4恢复，实际12更新/1344训练query；前2行父日志保留，3/4步任务/视频/query事件一致，
  optimizer步数、scheduler、sampler及world2 rank RNG资产完整。R_V原fresh4仍来自`0011d292`，恢复CPU窄修`4dde66be`不改变其模型。
- Source可训练参数0。reader首步活动；R_V Text/VL/Action/Core/Procedure从第2步有有限非零梯度，
  R_L只有Text/Core及reader活动。真实FM—memory cotangent—同版本教学encoder重放有实际路径，不证明语义或学习充分性。
- 最长fit视频task29/demo0为347原帧、stride5保留71帧；一次28query反传无update，峰值allocated20.56/reserved22.61GiB。
  13.38秒是整condition（含memory生成与重放），其中首次memory2.81秒，不能将两项相加当纯FM加memory总成本。
- 六条train-only接口case初态/body位置、env seed及共同policy-noise前缀配对；全部真实动作与T+1状态/谓词齐全，
  两条global2是双相机full。成功提前终止的一条有168实际动作/169状态；最后一个已计划5-action前缀剩2条未执行，
  按实际终止截断后与npz动作相同，这是正常capture语义，不是原件缺失或多执行。smoke成功数不作科学收益。
- 逐段完整GPU计费复算0.5125833小时≤0.75，包含加载和失败等待；等待段为保守上界。
  输出约680MiB、开发及冻结代码合计506MiB，全部新增data1且低于2GiB/768MiB。没有新增held/Test、formal630或KD。

**科学投入裁决。** §156及机制分析§6.1已确认，旧Video Functional共同学习过视频表示/执行读头，
其VL版本及原生续算诊断均提供相关负证据。当前C/P、独立q/v纠正与单reader目标有实际图差异，
但没有证据表明它们改变旧失败所涉及的共享获取问题；工程只验证可执行性，不能补足这个理由。
根据四步工程外推，630训练约2.7–3.6小时wall（world2），224条件memory6–12分钟，224评测40–55分钟（4GPU），
后继8.2–11.1 GPUh，加本工程约8.7–11.6，上沿超原第一阶段9 GPUh。这是有限样本估计，非长跑实测。

因此结束本Reader窗口，撤回未激活的630正式学习准备；不自动加预算、删必要对照、做更小低信息量screen或接蒸馏。
正式学习运行量为零，不能把它记成Reader性能阴性、普遍不可行证明或新方法实质成功。
临时994行/三个源码owner及测试由Git `2fbd4c7d`和原件保留；源码历史合入main但active tree退役该独立诊断路径，
不引入第二Writer/fallback。Source、S0/C0、checkpoint及原始六条case均保留。
源commit兼容例外仅认可本次核定的`0011d292→4dde66be`恢复，不授权任意未来实现沿用旧checkpoint。

完整目标继续推进。当前数学认识仍以真实联合特征、生成算子和跨episode学习如何形成执行作用为对象；
既不把旧可辨识性/条件独立推导重新包装为统一根因，也不因撤回Reader就立即拼另一版近似架构。

## 161. 公共路径的最近似历史边界与可预测功能更新（2026-09-26）

指定执行者完成只读核对，主讨论独立核对关键源码/配置与原始行为汇总，接受[机制分析§14.5](docs/analyses/feature_to_operator_mechanism_20260926.md#145-最近似共同路径历史已核清不能混为一个实验)的完整区分：
shared prior43/250来自固定零code rank12的特权功能蒸馏，随后冻结，phase-code rank4残差37/33且未实施合法RGB Writer；
Unified v3/v4冻结该carrier，真实视频表示与signed native pooling在跨episode FM下联训，common-base只是查询`b(C)`，
held5分别35→31、45→40；DJNFR只学冻结LPCP后的8个direct A/B head，因子相加有交叉项，成功轨迹信用后143→136/400。
不同面板不相减；未运行的cycle2、validation400或最终controls不写成失败。

所核对完整近邻未直接检验fresh自由常量完整LoRA与合法视频完整残差共同接受同一跨episode FM的精确组合。
这只是覆盖边界；旧结果仍削弱“公共能力+条件更新+真实梯度自动保持”的解释，不支持因未试组合再投。
主讨论§15进一步从实际LoRA的局部动作Jacobian定义条件风险：理想局部位移取决于可预测的功能梯度及其执行度量，
不是逐位置标签、裸参数差异或视频特征可分性。给出嵌套表示的局部风险差、条件均值回归分解及三种detach梯度。
这些是带明确条件的数学关系，不是EMBER信息不足/曲率问题的测量或闭环收益证明；旧LocalField已实际监督并消费cotangent，
不能把“直接学更新”重命名为新方案。本次无新增模型/环境运行，无统一根因、验证修复或新GPU合同。

## 162. 外部优化过程方法不能替代合法视频获取论证；进一步明确实际控制作用（2026-09-26）

执行者只读核对已验收，主讨论复核官方固定源码与LocalField实际梯度/收缩代码，见机制分析§15.5。
HyPoGen的条件是显式任务参数，内部是可学习伪更新，自由初值与生成器由动作MSE等共同学习；不是合法视频到冻结source LoRA。
HyperNet Fields在当前生成权重处用真梯度约束相邻权重，但原论文未指定detach/二阶实现，未找到官方源码；
不能把某个推导分支写成实做，亦不能从端点一次生成推出视频获取与能力保持。
旧LocalField已让收缩前真cotangent监督直接构成LoRA的U/r，不再以“直接学更新”新开近似方案。

本轮不激活这些训练形式，也不因条件Jacobian耦合自动改投自然梯度/白化/梯度匹配。
当前private Writer/freeAB、head缩步与任务lookahead的反例继续限制这一判断；不是所有优化方法的否定。
机制分析§16将教学作用具体写成原生Q的相对读取偏置、suffix V的动作位置通信及实际FM信用：
固定LoRA仍能根据新执行状态改变作用，前提是状态特征与原生Value提供可利用的区别。
注意力定位、非零梯度、过程标签与闭环能力不能互代；“状态条件规则”是解释模型，尚非已验证的表示语义或新架构。

## 163. 教学绑定的功能信用与跨初态识别：完整解释模型及其尚缺依据（2026-09-26）

主讨论自主理论接续见[机制分析§17](docs/analyses/feature_to_operator_mechanism_20260926.md#17-一个完整的解释模型动作信用怎样教会教学特征的使用)。
本项没有新增模型／环境／GPU计算，不是新的学习结果。旧有序控制理论与SemanticPath已经完整提出并实施过
角色／状态路径→LoRA→联合FM，故不能将这一描述本身算作新方法。

新增具体联系：固定两个竞争绑定的真实FM功能修正k1/k2，令教学以c=sigmoid(ell)选择，
则信用为`c(1-c) E[(k1-k2)^T e]`。若k1确为条件平均正确修正且其他残余与差异正交，
梯度为`-c(1-c)^2 E||k1-k2||^2`，会增加正确绑定。这解释动作监督怎样教视频特征的使用；
也说明可见语义可分但功能作用相同、其他作用尚错、或共同学习k时，不能保证同一方向。
它仍可能由语言／静态内容学成，不单独证明动态视频必要性。

固定特征的局部可达子模型进一步得到联合特征`Phi=m^T kron D`，同时包含教学条件与自身状态功能响应。
新分布函数是否由训练函数唯一确定，取决于相应Gram零空间的包含关系；有限估计误差还可能被放大。
二值角色／状态例子展示了语义完美、训练拟合完美但新初态行为仍不确定，不能以更多相同关系的episode消除歧义。
这些矩阵没有在EMBER测量；不是信息不足、支持不足或优化病因的宣布，也不授权曲率探针、补任务或预条件实验。

在teacher/query条件独立且参考功能不随teacher变化的进一步限制下，局部风险分成task平均教学特征的功能拟合，
与`tr(F Theta Cov(m) Theta^T)/2`。由此明确主FM怎样鼓励可复用控制信息、抑制没有query支持的功能波动；
它不要求裸特征相同，不新增一致性loss，也不把期望分析变成部署平均LoRA。旧§128的期望等价性边界继续保留。

完整链落实到实际38-target A/B梯度、Q/V作用及10步flow后执行5步再反馈；不把一般采样性质恢复为本项目研究分支。
SemanticPath、真实空间边际监督、局部逆动力学和LocalField各自已提供完整近似负例，
它们并非缺少最基本的可微性或动作信用。当前仍缺一项有依据、能改变这些失败预测的完整机制差异。
因此本阶段不激活新架构／辅助标签／Reader／公共底座或GPU合同；后继推理必须产出能改变方法判断的预测，不能继续堆公式。

## 164. 获取与保持不能作为无条件二分；不以“现成视频知识只差释放”启动新路线（2026-09-26）

主讨论继续理论裁决，见[机制分析§18](docs/analyses/feature_to_operator_mechanism_20260926.md#18-获取与保持的竞争解释可以裁决什么不能据此自动选什么)。
用明确的表示／编译局部模型比较缺失关系方向与已保留关系但输出映射失配；两者可有相同当前输出，
也可以在不同能力上同时发生。真实联合更新混合表示和Compiler的变化，不凭总分或梯度非零定位责任。

需区分输入有线索、表示可被学习利用、现有共享完整映射已具能力、后续保持四个层次。
真G和private任务适配增加了动作信息／学习，不能证明原合法RGB表示已经学好了控制知识。
当前C0相对B的27得／28失是方法间交换；同一D1050→1260的123→91、保留63／得28／失60才是时间损失。
旧固定读取／Compiler的共享D干预亦证明输出映射变化足以改变控制，但同一D还有训练侧正作用，不是普遍有害结论。

旧普通FM和P/Core正例削弱“完全不能学／完全没有功能信用”；SemanticPath、空间边际、LocalField、联合Reader、
private Writer/freeAB、H54及lookahead分别限制简单的表示补救与输出／更新补救，不能把一方阴性当成另一方确诊。
冻结E重新训练共享Compiler，只有在事前限定decoder类、学习算法及为何改变旧失败预测时，才形成有区分力的完整方法假设；
其成功首先说明E可用，未必证明新学习前已有控制器。交换跨checkpoint模块还可能破坏正常共适应，不自动定位遗忘位置。

**投入裁决：**不把“视频知识已经就绪，只差读取、保持或预条件”作为下一路线的既定前提，
也不转投没有新依据的encoder／语义标签修复。保持仍与获得新的可迁移视频作用同时验收。
本轮没有形成有实质机制差异的完整候选，无新实验／GPU／成本合同；不把理论边界误称为根因发现。
这条理论延长线收束，不再仅用同批证据自排同类公式／报告任务；整体目标未完成，研究授权未撤销。

## 165. 持续方法研究：输出限制的旧反证与参数条件回读的真实信用边界（2026-09-26）

Owner纠正停工后，主讨论继续完整方法研究，见机制分析§19–20；本条不是新实验结果或active设计。
当前共享B头具有共同216维输出列空间，不保证表示所有独立rank16常量LoRA；但这是早已记录的结构限制。
完整Target-Owned已解除跨层共享并实际形成异质方向，correct99/76/86/68；§126投影后固定16条件未出现净下降。
这两项反证阻止主讨论由同一限制再提出拆头、扩维或几何小扫，也不外推所有自由输出均无效。

正在审查的具体图是：第一次完整条件解释生成临时w1；裸source+w1读取同一合法教学RGB/语言的H与velocity；
原内容与这次响应共同生成唯一最终LoRA。没有任务内loss/optimizer、环境交互或第二部署adapter。
新增的w1信用是`J_R,w1^T J_U,R^T J_D,h2^T g_out`，并非w1自身的真实任务梯度。
因此回读不是自动知道对错，也不能把非零响应说成有益自我纠正；它至多改变条件读取与最终功能监督之间的计算联系。
回读不增加原始视频之外的信息，但可重新读取初次压缩没有保留的内容；该可能性尚未在EMBER验证。

若两次共用当前zero-init B头，在局部有界可微前提下，新图与固定identity回读的有效输出差起于B头尺度的二阶，
相应头梯度差可从一阶进入；精确算术下第一步相同。不是永久不动子空间，也不是实测学习延迟，不能据此扫初始化。
真正的比较必须保留相同第二次读取与h1直接路径，避免把Action Meta是否活动或额外深度混入参数条件化效应。
旧Reader吞吐不能充当该图的实际成本，当前没有工程/profile或GPU派发。

指定执行者已用约10.5分钟完成只读核定，主讨论复查关键外层安装和写回后验收。
`8553b61`在Writer前恢复identity；`184947cb`写回Z/H激活、最终参数在原生续算之后生成；
Process–Composer/Axial/Unified/Horizon同样没有本次task参数返回教学native forward。
这是所核定完整图的拓扑区别，不是穷尽历史或新方法有效的证明。
当前不让“仅参数回读+末端FM”进入工程：它没有说明为何能改变最近Unified条件化原生续算的失败预测，
也未把内部w1变成有真实纠正语义的策略。后继研究完整纠正学习的合法特征、当前参数处标签和实际消费者，
不把§15旧梯度路径约束、LocalField与新读取机械拼接成新候选。
一条候选缺乏依据时关闭的是该候选，不能再次让整个未完成任务停在报告之后；实际承接状态只看progress。

## 166. 当前参数纠正与真实视频知识是不同学习要求（2026-09-27）

主讨论继续完成[机制分析§21](docs/analyses/feature_to_operator_mechanism_20260926.md#21-当前参数处的纠正什么时候是学习算子什么时候只是重写fm)，
没有新模型计算或实验。完整待审图为合法条件→u1→教学回读→预测因子增量→唯一u2，
训练标签为授权跨episode查询在u1处的真FM梯度；标签不进入部署输入。

停止梯度的目标端点回归与增量回归虽有相同数值残差，梯度分别为`J2^T r`与`(J2-J1)^T r`。
前者在两端点和Jacobian相同时只是中间普通FM信用；一般情况下还改变局部曲率和信用路径，不能称新增了动作知识。
二次模型`g(u)=Fu-b`说明当前参考点可以修正Fu依赖，却没有自动提供视频如何确定b的机制。
不能用反复施加源点梯度的反例解释旧LocalField，它并未做这种反复更新。

只用线性化教学响应Pu识别执行梯度，需要`ker(P)⊆ker(F)`；真实EMBER尚未测定此条件。
完整图还输入u/h1，故该条件不是完整图的不可学习定理；它限定“全部纠正由回读获得”的解释。
同一有效BA的因子缩放改变原始A/B梯度与SGD作用，当前梯度标签并非天然固定的控制坐标。
旧LocalField使用固定裸source的物理输出坐标；改变为当前参数完整FM还同时改变查询state/flow、坐标及信用，不能称单变量。

明确的相对梯度误差和步长条件只保证同条件局部FM下降，不保证未见任务上预测成立、跨共享更新保持或实际闭环成功。
本阶段不启动当前参数回读＋梯度回归，也不为其追加坐标/步长/轮数修补。
这些结果把该路线的作用限定为参数依赖和训练信用组织，尚未构成合法视频控制知识获取的修复；
后继转向观测到的对象对应/关系变化怎样进入完整控制规则，而非继续重排优化公式。

## 167. 教学关系与自身反馈的具体分工，以及尚未建立的原生联系（2026-09-27）

主讨论完成[机制分析§22](docs/analyses/feature_to_operator_mechanism_20260926.md#22-从看见移动到状态反馈规则一个具体联系及它缺少的条件)。
只读合法fit task2/demo0的12张agentview RGB作例示，没有teacher action/state/reward读取或模型/环境执行。
多个相似碗、手部遮挡和目标相对盘子的移动说明角色连续绑定与关系变化的具体含义；没有从12帧推定接触/释放标签。

联合patch对应与两端空间边际不同：相同边际可有相反交叉矩，平均位移也可相同。该反例不证明完整Horizon不能隐式表示对应，
更不证明原生相似度就是物理跟踪。旧可识别性分析§7已明确审查过这一输入，并未实施；因此不将它作为新发现或失败实验。

解释用模型`H(x)=Σ s_i p_i^T`将当前物体语义和坐标结合，合法教学给出角色/关系，自身观察决定位置与模式。
在明示的固定特征条件下，任务控制差能写成固定低秩`K A(m)`，并保留固定模式中的自身位置反馈；
跨初态动作信用通过`H K^T K H^T`约束同一个角色解，不要求各样本梯度同向。
这给出了“视频知识不等于教师动作复制，静态LoRA仍可形成状态反馈”的可算联系，未证明原生hidden具有这些坐标或身份记忆。

真实Q/V/38处LoRA仍需学成角色—坐标—模式联合特征的功能读出；末端FM按执行差异分配信用，并不会自动标注正确物理对应。
同样轨迹拟合可对应横向稳定或不稳定的反馈，rank1变化也能翻转稳定性；这不是EMBER已定位的不稳定病因。
当前跨episode训练已补充状态覆盖，不能当作从未存在；旧LocalField和联合Reader也已有真正执行消费者。

裁决：不启动“追加显式运动/联合对应，再接旧Compiler与FM”，也不为其排辅助标签或模块关卡。
下一项依据来自当前C0/B既有得失行为，而不是旧Goal大缺口：一次8对、四suite各一得一失的只读描述性分析，
让完整方法同时解释目标绑定及之后的控制/关系保持。按结果选择的案例不估计失败比例，也不定位唯一模块根因。

## 168. 当前得失原件限制“认错对象”解释；同C0换正确视频改变拾取进展（2026-09-27）

固定8对已存C0/B轨迹只读分析12.3分钟完成，主讨论核读报告并复算Object(14,10)、Long两对关键数组。
完整范围与解释见[机制分析§23](docs/analyses/feature_to_operator_mechanism_20260926.md#23-当前得失行为同一正确对象的控制差异与正确视频之间的变化)。
这是四suite各一得一失的按结果选择，不能估计总体失败类别比例；没有模型/环境/GPU或新评测。

Spatial两对、Object两对及Goal21/8的失败臂都移动了目标对象而未完成关系；Long涉及第二子目标未覆盖、
以及成功轨迹里第一个子目标短暂false后恢复。Goal20/43没有抽屉joint/把手/full图像，只能保留无法细分。
这削弱全部归因持续搬错对象的故事，不证明内部角色/参照绑定已经正确，也不定位唯一模块。
body原点距离不是接触或区域判断；成功即终止，不证明终止后保持。

主讨论对已固定的唯一full案例14/10，补读同C0、同state10的既存other轨迹。
demo20/37条件分别失败280/成功173，首个双相机/语言/state/环境初值及noise共同前缀相同；两次搬动的都是ketchup。
高于初始body原点30mm的描述性代理首次在263/104步发生；不是grasp标签。首次replan前5动作已经不同，
5步后EEF差1.30mm，之后反馈也不同；因此有真实视频条件作用，却不能由此证明语义正确、参数差小或总体视频净益。
其它7对的Source/other已有成败一并记录，不挑视频、拼checkpoint或扩面板。

完整方法须解释对象/部位与关系知识如何形成可由自身状态重新判断的接近、实际带动、放置和未完成子目标控制。
不由这些案例启动识别模块、相对坐标、单调阶段、保持loss或多K；旧相对几何和多视频负例继续有效。
下一完整判断审查“观察到的视觉效果能否提供合法纠正信号”，区别于已关闭的纯参数回读；
动作条件效果模型须解释真实标签、可识别性与消费者，不能仅增加世界模型或预测loss。

## 169. 可见效果参照不同于纯回读，但不自动提供跨初态纠正（2026-09-27）

主讨论完成[机制分析§24](docs/analyses/feature_to_operator_mechanism_20260926.md#24-观察到的变化能否成为当前策略的纠正效果残差的完整图与边界)，
完整审查合法视频→临时完整LoRA→无state动作预测→前向视觉效果残差→唯一最终LoRA→跨episode真实FM。
仅数学/源码/已验收证据，没有模型、环境、GPU、数据构建或新审计任务。

当前H是t=1公共噪声上的hidden，不是10步flow生成动作；无state预测也不等于真实policy在teacher物理状态的行为。
可见下一帧给了独立参照，但把残差作为Compiler特征的末端梯度，不等于对该残差实施真实动作纠正。

精确局部效果模型给出的方向是`JᵀWJ(a−a*)`，有零空间且不覆盖所有接触/夹爪切换。
更细的正面边界是：模型即使没有真实动力学导数，只要在a*精确拟合，足够近处仍可提供局部拉回；
不能由导数未识别直接证明这一目标无用。相反，在有限距离、偏置或多根处，演示拟合准确也允许纠正走向错误动作。
专家动作支持上的精确等价反例、单维假根例子，以及不充分视觉状态下观察/干预分布的区别均已给出。
这区分了观测变化、局部动作约束、反事实效果和跨初态反馈规则，不把它们混成“学会运动”。

LocalActionGrounded、NativeCorrection/LocalField、24,480动作的真实物理J及V-JEPA近邻完整比较后，
当前组合仍未给出足以修订共享纠正获取负例的依据，且新增因果预测前提，因此不进入工程。
不先训练效果模型、采新干预或拆局部Gate；不把旧.29736/.25的non-pass改判，也不外推所有世界模型无效。
这关闭一个完整投入方向，不构成统一根因、修复证明或整体研究完成。

## 170. 原生attention信用给出有效参数方向，但换坐标不等于学到操作知识（2026-09-27）

[机制分析§25](docs/analyses/feature_to_operator_mechanism_20260926.md#25-把教学特征写到真实attention方向坐标成立还缺什么学习联系)
核清旧Semantic-Address的Writer query、Unified的native X/Y与PNBTT的learned candidate keys，都不同于实际source attention的prefix K。
当前源码的真实Q/K投影、RoPE、GQA及prefix有效长度位置合同已复读；没有模型或环境执行。

实际logit信用`c_ij=α_ij b_iᵀ(v_j−o_i)`，Q矩阵梯度为`Σ c_ij(R_iᵀk̃_j/√d)h_iᵀ`。
这比同shape多给出了一条真实特征到参数方向的关系；但c包含最终动作误差及下游计算，并非对象或阶段标签。
在合法native量固定的条件下，梯度是`M_z c`，其条件期望仍为`M_z E[c|z]`；换坐标没有新部署信息。
裸系数MSE还改变功能加权、约束无效零空间。单相机256patch的8-head系数已有2048项/query，不能声称天然降维。

低秩系数并不在有query位置RoPE时自动成为同rank参数；分别pool两端又会产生跨位置项。
复制teacher key到B也须证明它在自己观察的目标/干扰key之间形成正确margin，并由自身状态激活。
两维反例说明同维或同一对象称呼不足以保证这一点，不声称真实source已经出现该反例。

因此不实施“原生key池＋attention信用＋旧Compiler”，也不派发方向/语义head探针。
潜在归纳偏置优势保留为假设，不能把旧LocalField误记为独立辅助头，或只因参数方向更直观就重开相似路线。

## 171. 当前固定教学对含不同再接近过程，但不能解释成坏视频或自身episode复制（2026-09-27）

主讨论只读既定Object(14,10)两条教学demo20/37的完整stride5及末帧agentview RGB，49/33张，
没有teacher动作/state/reward、模型或环境；[机制分析§26](docs/analyses/feature_to_operator_mechanism_20260926.md#26-当前两条教学真正显示了什么成功示范也可包含再接近)
保留精确来源、可见帧及判断边界。

demo20可见瓶升起后与夹爪分开，再接近/带动后运入basket；demo37在采样帧中呈较连续的一次携物转移。
RGB不能分辨滑脱原因或主动调整；首帧也不同，不能把已有C0换视频成败单独归因于动态过程或顺序。
同一demo20条件在已有state34成功136，state10失败280；demo37在state10/12成功173/213。
四行仅按这两个固定condition查既存索引，不扩评、不挑视频，也不宣称完整crossed实验或视频质量统计。

frozen `dca1b550`配置及实际分支确认C0两组21+7 queries均跨episode、均完整horizon普通FM；
第二组`extra_endpoint_prefix=false`，保留的tau1/前5步字段并未生效。
所以不能从教学的可见调整倒推“同视频辅助使C0复制自身轨迹”；旧同视频165/147有限正增与后续158/159也继续保留。

这使教学中的可复用知识更具体：对象/关系及动作适用条件，应使策略在自身情形下使用接近、带动、运输或恢复，
不能复制固定次数/时长，也不能将恢复片段一概删作噪声。尚未证明source hidden或当前Writer已学成这些功能。
不由此开启阶段head、时间不变loss、视频过滤或旧辅助目标重测。

## 172. 同视频信用同时改变联合功能风险与实例信息，已有正例不授权恢复旧辅助（2026-09-27）

[机制分析§27](docs/analyses/feature_to_operator_mechanism_20260926.md#27-同视频功能信用能教什么运动对应实例信息与跨初态反馈)
核清三种合同：旧24-task及后来36-task的匹配实验只改变端点/前5步辅助的episode配对；
当前C0两组均为跨episode完整FM。旧辅助query有其自己的真实图像/state，误差经最终完整LoRA与主项共同反传，
不是独立局部头，也不是把state输入部署Writer。

固定参考的局部风险中，同视频不只改变b=E[Dᵀr mᵀ]，还改变H=E[(mmᵀ)⊗(DᵀD)]；
只补一个标签协方差不是完整目标。匹配驻点差异满足
(H_main+λH_same)(θ_same−θ_cross)=λ(Δb−ΔHθ_cross)，在有唯一稳定解的相关子空间内成立。
D是真实query自己的state/noise/flow及原生续算的功能导数，不能换成teacher hidden。
该固定表示局部关系不是实际Adam或非线性共同学习的已验证解释。

τ=1把数值动作插值从query去掉，但同视频的未来RGB仍能提供实例动作信息。
不受LoRA函数类限制的最优端点动作回归为E[a|q,C]，它不等于十步flow采样或当前生成器的可达保证；
paired误差降低既可能来自共有运动—动作知识，
也可能来自布局/路线匹配。两种解都能利用自己的state和同一LoRA，不能以“真实消费者已接通”排除后一种。
不同成功路线不必有相同LoRA/动作，合法恢复片段也不能一概当噪声。

两份匹配历史一起约束：旧1500 same/cross=165/147、2100=158/159；
覆盖重划后的cross−same六节点为+41/+28/+39/+50/+12/+44，见
[36-task原报告](docs/review_materials/20260922/auxiliary_pairing_fresh/report.md)。
前者保留有限正增，后者削弱“最终LoRA接同视频就会恢复”的充分性；不能跨合同唯一归因任务数、数据关系或优化。
当前纯跨episode C0=120、语言121说明删去旧辅助也不是当前问题的充分修复。
本次未使用最终shuffled/reversed controls修订设计。

完整原则为：教学应辨识可迁移控制关系，由同一生成LoRA在query自身状态下实际消费；
跨episode风险为主，同视频对应只有促进这层映射才有价值。当前主项已有该风险，
缺的是有限共享学习怎样取得并维持这种关系，不能把原则重复当新方法。
LocalActionGrounded、NativeCorrection/LocalField、旧联合Reader与旧最终LoRA同视频监督均纳入最近完整比较。
没有形成足以改变这些近邻失败预测的新候选；拒绝旧辅助/权重/采样/坐标的重组，不派工程或虚构成本。
本条配对风险理论线结束，不再自排同类公式任务；整体研究目标没有完成或暂停。

## 173. 视觉控制的作用映射不能由点轨迹或实际A语义监督替代（2026-09-27）

[机制分析§28](docs/analyses/feature_to_operator_mechanism_20260926.md#28-视觉对应何时成为控制必须分开三个导数与旧语义rank提案)
定向核读RoboTAP/Track2Act原始方法，保留其相机/几何/机器人数据与部署控制依赖，不移植外部分数。
旧可识别性§7已经审查跟踪特征，本次不是首次发现运动对应，也不重新启用跟踪器加旧Compiler/FM。
腕部图像Jacobian可在接触前提供控制联系，不等于固定外部相机下的物体效果Jacobian；
后者又不同于真实FM/十步采样对LoRA的导数。简单视觉误差下降要求实际作用映射在可控方向上合适，
准确点对应、真cotangent和物理坐标均不能单独提供这一条件。

查回9月13日操作语义分析：它已经提出监督执行query的实际A响应，而非只监督独立head，
并明确零B、rank抵消、共享A不需视频的反例；本轮U M(C) S重分解没有解决它们。
原接触/region标签限制也保留，不将这条旧未实施结构改名成新候选。
新判断把教学参考/关系、自己状态特征与控制增益分开：前两者准确仍须经有效动作映射，
但正确控制先验若真实存在，确可减少从视频推断完整动作的负担；这是条件性原理，不是当前能力事实。
无模型/GPU/环境/新数据，未形成新的启动合同；主讨论在同turn继续审查完整的最终速度投影编译与共同执行特征学习。

## 174. 条件速度场可精确编译，允许有界工程但尚无能力证据（2026-09-27）

[机制分析§29](docs/analyses/feature_to_operator_mechanism_20260926.md#29-共同学习状态反馈基直接编译条件速度场一个有明确代价的完整候选)
与[候选合同](docs/designs/conditional_velocity_operator_design.md)选择一个完整受限模型：
source冻结，fresh自由共同rank128 LoRA β产生自身状态h，U h提供共同状态函数，合法视频生成R[7,256]，
在原生末端增加E7 R U h。β/U/R/encoder只由同一跨episode完整50-horizon真实FM共同学习。
R到固定自身query的FM输出为全局线性；其梯度为真实七维残差与状态函数的外积，公共actor也受同一误差训练。
教学不接query/标签，执行不接视频memory。没有辅助教师、几何语义loss或裸source纠正标签。

在action_out拼接[Aβ;RU]/[Bβ,E7]，其它37目标零补齐，得到唯一完整rank135 LoRA，精确实现该场。
共同rank128解是可达子类，非训练轨迹或保持保证；条件部分只改最后投影，主动承担早期信息须由共同特征保留的限制。
给定FM query时h独立于视频；十步采样中后续latent会受视频影响，不能把整个采样/闭环误说成线性。

旧Video Functional的执行query来自未挂LoRA且无梯度的source，另有LoRA学生；新图共同学习执行特征且不需近似传递。
LocalField的同参数监督、旧冻结公共prior/Unified/DJNFR负例及原完整Writer/private正例均保留，
不把少输出参数、短梯度、常量可达性或“精确组合未试”单独当立项理由，更不宣布冻结query为统一根因。
候选原理是共同状态反馈函数与教学系数的可迁移组合；获取、作用覆盖、保持和强MT-BC之外的视频收益均待测。

仅授权指定执行者有界工程：6宏步/672query、一次长视频28-query反传无更新、2条train-only接口episode，
最多2GPU/0.75完整GPUh，data1原件峰值4GiB/代码768MiB；预计60–90分钟、120分钟工程判断上限。
没有正式学习、held、语言第二臂或旧实验恢复许可；实测成本返回后主讨论另冻完整比较。
若完整有信息量学习窗口无能力/视频收益，关闭该条件速度场组合，不逐层/逐rank/辅助配方扩展同一提案。

## 175. 新候选的分支消融有公共分解歧义，完整比较须用训练语言参照（2026-09-27）

[机制分析§30](docs/analyses/feature_to_operator_mechanism_20260926.md#30-怎样识别这个候选的视频收益公共分解的歧义与完整比较)
说明`W+E7 R(C) U`在算子层可把一个常量`E7 D U`移入W而从R扣除，完整policy不变，
但“令R=0后的policy”改变。这个算子分解不声称固定超网络可简单平移到任意D；
它与可逆R/U坐标自由度共同限制解释：非零R、其norm和去R掉分不能单独认证视频贡献或公共能力保持。
不为本候选添加内部置零/几何诊断；后继以完整正确/另一正确视频、独立训练语言参照与强MT-BC裁决。
旧FactorHeads也已有可学A侧投影，U不是首次学习状态地址；新假设是整个条件速度场及共同执行特征的分解。

本次定向读取强MT-BC selection及原run_contract/results，确认coverage协议step300=155/400、raw source1000、rank128、
Validation IDs `[3,6,11,16,23,26,31,39]`、每task50初态、seed7及官方flow/预处理。
这与C0 fit28/diagnostic-held8的120/132/语言121不同，不能相减定位原因。旧C600154/other160只作其真实完整合同下的参照。
新V/L须明确相同纯FM事件流/尺度，不能继承旧21+7辅助的隐藏4/3或端点分支；正式节点和预算尚未冻结。

原evaluator已经支持不同LoRA的批量执行，不新增batch backend。rank135的完整FP32 bank每条件约41.43MiB、
400份约16.18GiB；正式物化须单独计容量，可用共同β/U与每条件R精确存储，rollout前仍构造一套完整LoRA。
这是后继实现/成本边界，未扩大当前4GiB工程或启动物化。历史约一GPUh的400-panel调度仅作成本锚点，不冒充新profile。

## 176. 条件速度算子工程接通，下一证据必须来自完整V/L学习（2026-09-27）

执行者隔离commit `505590808f744f3b5d30da0e80c0e645841d282c`已完成工程；主讨论独立读合成、
实际FM与VJP、采样/恢复源码，并核原metrics、查询前后缀与两条NPZ。38目标/76因子以128+7拼接，
实际action_out为公共算子加E7 R U，其余37目标附加零算子；临时完整LoRA的cotangent正常回到共同β和条件路径。
四task等权后跨rank SUM，没有将共享β重复计权；source物理trainable0，读取教学时不装公共β。
首步公共β及零head有信用、第2步U/Text/VL/Action/Core/P有信用符合identity初始化，不能据此证明学会视频操作规则。

真实6宏步/672query，恢复checkpoint2→4只新增两步；父日志4行保留，子前2行与父对应前缀相同，后2步query trace对应。
global29/demo0的347raw→71帧28query反传无update；global2/32各一条train-only接口失败220/520步，真实T动作/T+1状态齐全。
四次启动exit0，完整0.1269607541 GPUh、峰值2卡；study0.737GiB、代码493MiB。原件根为
`/data1/user/ymdai/ember_runs/conditional_velocity_operator_engineering_20260927`，成本和GPU原件未写data0。
工程19:40–20:16 UTC约36分钟，原估60–90分钟。消息Queue `01a0df5c-5fee-7361-8c9a-534997a96404`已由主讨论消费，迟到不重跑。

实测宏步16.822秒/112query，最长profile14.07秒/21.46GiB reserved。新36-task全50教学的manifest长度
P50/P90/P99/max为29/51/87/105采样帧，最大raw517，故71帧工程profile不覆盖新协议最长视频；这是已识别的成本边界，
没有补开模型probe。强MT-BC完整validation曲线50..500为82/112/99/133/136/155/131/130/138/119，
说明早期小幅波动不能被改名成架构故障；也不保证本候选同update/不同query曝光会重复该曲线。

设计§6–9冻结完整V/L比较：同一coverage36、纯28-query跨episodeFM、相同3e-4/150warmup/1200cosine时钟、
共同参数匹配初始化；首批各270更新/30,240query，再各official400，下一预期450节点不自动启动。
首批总硬限9完整GPUh、data1峰值8GiB，紧凑bank只改变存储。当前只派CPU实现/集成，GPU仍需明确冻结commit启动。
直接完整比较是为检验共同状态反馈基与视频系数能否学到有益作用，不增加模块Gate，不用R置零或loss替代视频收益。
当前没有能力、稳定性或视频必要增量的新结论，Reader不恢复。

补充机制分析§30.5的精确容量边界：36个任意固定7×1024末端矩阵可堆成252行，均可包含于共同256维行空间。
所以U的宽度不排除记忆已见任务的固定输出算子，不独自保证可迁移运动规则；这不保证encoder能学出该分解，
也不涵盖任意视频条件或完整policy。冻结V/L/held比较保持不变，不据此另加瓶颈/正则/任务识别实验。

## 177. 完整V/L实现验收与事前合同纠正（2026-09-27）

指定执行者以7f5374a7交付完整V/L训练、事件、紧凑bank及官方评测接入，纯CPU约42分钟，无模型/环境运行。
主讨论独立读实际图/采样/FM/VJP/部署/配对接口后纠正两处实现：自写LR在warmup后又衰减1200步；
L第二Value取了Core前的text。最终18b7e4b7直接复用强MT-BC时钟owner，并两次消费同一真实Core语言memory。
这些修改在任何正式学习/评测前完成，恢复预注册合同，不是根据科学负分改架构/超参。相关22项CPU检查通过。

主讨论fast-forward集成main；图/合成、36-task数据、ECP训练、bank/官方adapter各一owner。
旧generic CLI与工程case/profile出口退役；native packing及selection/file/source共享内部依赖暂留，
主讨论在本候选首次完整裁决时处理其保留/退役，不增加旧Compiler fallback或第二评测平台。
GPU许可仅限设计§6–10首批两臂各270更新/各correct400，完整9 GPUh；具体冻结/派发/运行状态见progress。

扩展Source-SFT旧authority测试失败在集成前main2a34b5dd被独立复现，相关旧配置/loader无本轮变化；
新训练仅借用纯scheduler函数，不调用该旧MT-BC配置loader。这是未解决的既有维护问题，不能说全库测试通过，
也不由此改写历史基准、重跑旧实验或宣布新路径科学失败。当前仍无新闭环能力/稳定性/视频增量结论。

## 178. 条件速度算子首点具备较强能力，尚无稳定视频增益或保持（2026-09-27）

冻结0c4ea636，V/L各270更新/30,240query和各official400已完成；主讨论独立核全部1080个匹配训练事件、
800行配对、bank编排及16份full PT/NPZ的实际T/T+1与初始状态。V151、L147、强MT155、Source51。
按Spatial/Object/Goal/Long，V为59/44/26/22，L为58/42/28/19，MT为52/47/36/20；三者广度6/8。
V对L R/G/L=124/27/23、churn50；V对MT=113/38/42、churn80；L对MT=105/42/50。
两臂对Source都只保留30/51，分别新增121/117、丢失21。Goal26与Object16的损失在V/L同时存在。

这是共同学习自身状态基与可精确编译条件场的有限完整能力正例，不能推出旧Writer不可学习或已定位统一根因。
同时削弱“固定query线性+共同特征+精确编译自然足以学好视频”的强解释：首点视频净增只有4，四suite差额均小。
不能把L147归给beta单支，也不以R非零/置零或内部norm认证视频；两完整模型不是纯信息干预。
30,240对历史172,800query的比较也不证明架构样本效率，因为宏步组成和warmup进度不同。
具体公式、每task表与边界见[机制分析§31](docs/analyses/feature_to_operator_mechanism_20260926.md#31-第一份完整vl证据能力形成视频增量与保持尚未成立)。

首点不通过最终资格。主讨论选择只完成原预留450节点、全50/50教学覆盖，不改图/配方；
若第二点仍主要V≈L且没有改变判断的完整正面视频证据，不以共同能力高维持视频路线或继续局部修补。
设计§11的CPU续训登记已完成，79c36003经主讨论独立审阅及7项CPU检查合入main；§12给出有界450正式范围。
实际GPU恢复和新分数仍待执行，CPU通过不提升科学结论；不运行旧实验/controls/RL。
内部共享依赖继续仅为当前候选服务，450保留/关闭时裁决退休；旧generic CLI/Compiler不恢复。

七次launch含一次漏设LIBERO资产路径的2.36秒prepare失败，未产生episode；补环境后同冻结代码完成。
完整5.703205/9 GPUh、峰值3卡、首末wall2.670h；study终值2.674GiB、代码506.78MiB，最终双节点无本执行者GPU进程。
没有连续CUDA显存高水位记录，不拿工程profile或no-OOM冒充本轮实测峰值。
原件根`/data1/user/ymdai/ember_runs/conditional_velocity_operator_learning_20260927`，
完整账目/机械报告在`launch/resource_accounting.json`与`analysis/stage1_mechanical_report.md`，主讨论复核见机制§31。
完成Queue `01a0e03d-5a82-7bb3-b916-edc365ff2e16`已从原生完成事件核定并消费；迟到消息不重执行。

## 179. 条件速度场450两臂共同退化，完整候选关闭（2026-09-27）

后续原件勘误见§186：L450在global39的24行fixture初位不一致，故涉及L450的相应严格配对范围为376；
这24行均双臂失败，总分与R/G/L数量不变。V270/V450与V270/L270的400行已存初位一致，关闭裁决不变。

冻结cfd378b6，两臂从270各新增180更新/20,160query，累计50,400、36task各50条不同教学；事件/时钟匹配，source冻结。
主讨论已独立核训练/恢复/bank、完整400配对/成功集合与固定full，补齐L450全部400实际NPZ的T/T+1及8full初态比较。
V151→102、L147→112，强MT155/Source51；相邻R/G/L分别79/23/72与89/23/58，V450对L450为84/18/28。
两臂Spatial、Goal、Long下降，Object上升；广度6/8不变。具体表、行为边界与原件见机制分析§32。
六GPU段全exit0、无失败重试，4.521025/6 GPUh、约2.238h；两批正式累计10.224230 GPUh，data1 study5.015GiB。

精确编译/实际梯度事实保留，未兑现的是稳定视频增量和能力保持的完整学习假设。
语言臂共同下降限制视频独有根因解释，但不唯一定位优化器/干扰/过拟合；当前没有训练侧闭环新证据，不移植旧run结论。
270正例未先证明好视频知识已获得，450下降也不能据此认定只需保持修复。常量解可达仍不保证有限训练轨迹。
关闭该组合的更长续训、分支补丁和controls投入；不宣布全部视频生成或共同状态基不可能。
必要CPU运行面退役见候选设计§13；两个冻结树、270/450依赖与原始证据保留。当前无active GPU设计。
Owner恢复自主研究；执行者完成一批后一次异步回报，主讨论主动接续，不持续等待、重复通知或制造空任务。

## 180. 函数表示不替代视觉变化到控制作用的学习；教师函数也不是必须完整重建的目标（2026-09-27）

机制分析§33核对Function Encoders/Basis-to-Basis的真实样本要求：输入函数样本可编码、基与系数可学习，
不等于action-hidden RGB已给出控制函数样本。视觉转移到策略系数的自由回归仍承担旧Writer的核心学习问题。
在解释用已知线性模型y=A_V c、b=B_Q c中，精确转换要求ker(A_V)包含于ker(B_Q)，稳定转换还需适当放大界；
这不是对真实RGB/完整policy已经核定的模型。增加帧不自动补齐未知作用方向，也不由此启动覆盖探针。

这项条件针对还原指定函数，比任务成功更强。相同教学轨迹可对应多个稳定控制器及不稳定控制器；
共享先验可选到有用反馈，无须恢复教师恰好采用的增益。反之，重建可见变化不能代替这种选择机制。
不把该边界变成EMBER不可能性，亦不把确定性动作回归当作真实FM的完整说明。
对照LocalActionGrounded、LocalField、Program/Unified及已关闭条件速度，当前提案没有改变完整失败预测，故不立项。
没有新模型/GPU/环境、部署求解器或标签工程；这是投入与目标边界的澄清，不是新的实证根因或已验证修复。

## 181. 条件速度运行面完成退役，历史证据读取保留（2026-09-27）

执行者5587c3df集成为main71ab5db5，13文件+202/−1338；专用训练/物化CLI、operator/data/trainer与三份私有规格退役。
只读bank模块移除训练/生成依赖，保留官方evaluator实际使用的唯一rank135重建和配对/capture身份合同；无新fallback。
主讨论集成后18项CPU检查通过，并对四套V/L×270/450真实bank的冻结spec、checkpoint游标及捕获合同独立核验。
两冻结树0c4ea636/cfd378b6和study/checkpoint/原件不动；没有新模型/GPU/环境或全权重重放。
这是科学关闭后的工程收尾，不增加性能证据，也不将候选失败改写为源码错误。当前无active GPU合同。

## 182. DISC支持功能监督训练参数生成，但不证明视频迁移或结构性保持（2026-09-27）

[机制分析§34](docs/analyses/feature_to_operator_mechanism_20260926.md#34-一个外部参数生成正例的边界disc没有消除条件获取与共享学习问题)
核对原文及官方固定commit4979e151。已见LIBERO-90高分、新任务带动作微调、未微调低分分开；不与EMBER分数直比。
实际图是语言→权重tokens/学习的伪更新→逐元素乘共同可训练s→完整MLP；执行视觉/state也共同学习，真实信用为动作MSE。
伪更新不读取实际动作误差；公开num_layers1配置的early_sup只监督一个更新后策略，不虚构多阶段真实梯度指导。
明确推导其共同s/生成器的跨条件更新仍耦合，常量生成器仍可使用场景线索；完整生成不在结构上保证条件有益或保持。
这保留普通功能监督的正面依据，但不补齐合法教学如何约束未见任务反馈函数的联系。
与旧HyPoGen/Unified/SemanticPath/LocalField及条件速度比较后，不采用伪梯度迭代的移植方案；无新工程或GPU。
仅只读公开文本，未运行外部模型或复现其结果；本次是方法适用性裁决，不是EMBER新实证根因，文献线收束。

## 183. 多参考纠正未补出新投入理由；整条反馈规则与逐点动作拟合的边界（2026-09-27）

机制分析§21.7补足独立多参考的局部风险分解：中心化参考新增F方向约束，源点已给出的b不因此多一个识别方程。
非线性与有限共享学习仍可能受益，故不是普遍不可能性；当前理由仍未超出§21，不开参考点扫描或新工程。
旧LocalField有完整双向视频、动作前后信息及同一U/r消费者，旧phase/SEOD有学生环境状态/真实十步监督；不能重复声称缺失。

机制分析§35给出跨决策联合行为的明确反例：两套两步必成功的反馈规则，按episode/位置均匀汇成逐状态动作分布，
每次独立重采的正确分布只有15/16在预算内成功。它不依赖平均动作；有适当时间/历史或分支状态区分时可以消失。
固定LoRA已有承载反馈选择的能力空间，不由反例自动增加隐变量；该通用现象也不解释EMBER相对MT-BC的差额。

随后只读预先固定train global34/38各前四条合法双视角RGB，全部stride5及末帧；没有动作/state/reward标签读取或模型/环境。
34四条均白杯先、黄白杯后；38四条均较靠近炉面的壶先、较远的壶后。时长/接近差异不直接等于不同可迁移控制模式。
没有据这8条得到搬运次序分叉的正证据，不扩大样本追逐故事，不建风格簇/重配数据或新训练。
来源与观察在`.codex/tmp/controller_choice_review_20260927/`，约4.4MiB，data1；不推全库比例、内部根因或新方法收益。

下一完整方法问题是允许多个成功反馈解的功能监督，须对照旧set-valued critic、功能code获取及SEOD/GOMQ，
不能用逐状态选专家或代码距离替代一套跨状态有效的规则。仍无active GPU设计，研究目标未完成。

## 184. 集合风险的教师一致性是真区别，但当前不构成新Writer投入理由（2026-09-27）

机制分析§36完成整体函数min与逐点min的判断。每batch统一教师仍只约束该batch，
`E_batch min`不能写成`min E_batch`；有限支持的函数一致又不自动给出闭环覆盖和成功。
真实十步端点信用经过完整采样Jacobian；普通FM更换动作标签同时改变插值输入，不能冒称固定query上的教师集合。
标签和教师选择只在loss侧，不能将action/state后验码输入Compiler后称为合法生成的主FM。

源码核定旧R2已经在四family上共同选完整教师成员，以detach权重监督残差更新方向，scale不接此项信用；
它不是逐层拼接，也不等于功能风险。R2整体弱与部分family改善、R10真实功能跃升和task-held不足均保留。
SEOD/GOMQ已有真实十步端点的收益与未保持，phase已有共同学生状态/latent处的expert velocity，不能虚构这些缺失。

成功轨迹不是可在共同状态上查询的完整教师函数；当前也没有可直接使用的全任务强教师集合。
若所有同task视频共享可接受集合，language条件的固定教师同样可最小化风险，新增min尚未约束视频到反馈的具体关系。
因此关闭“原Writer+集合端点蒸馏”方向，无新工程/模型/环境/GPU，不开温度/教师/panel扫描；不是证明所有集合目标无效。
原始依据、实际梯度和最近完整方法比较在§36。本轮继续审查完整方法决策，不把该关闭当整体停止。

## 185. 反例不是失败定理；新判断转向教学与跨初态查询的功能对应（2026-09-27）

机制分析§37修正了一个投入论证风险：实际A语义准确而B可以为零，只解除行为收益的充分性保证。
在简化联合损失中，零B处梯度为`-2 E[r f^T]`，相关性非零即有真实动作信用；不能据零B反例认定联合学习无效。
VisibleObject的有限正增、LocalField完整阴性和旧G2已有过程/谓词标签继续保留；未实施语义rank不记为实证失败。

新的具体问题是监督相容性：从某条教学抽出的相对操作参考，未必与另一条独立演示的动作标签相容。
§37.3用可算清的条件高斯FM说明，同一反馈参考下的不同初态查询可以改变条件最优场；
独立参考则仍需学习混合场。不把t=1均值误当作全部FM行为，也不由此认定当前存在多模式根因。

已形成具体候选合同`docs/designs/demonstration_transfer_learning_design.md`：仅训练侧用一个完整教学参考构造
不同初态的真实查询轨迹，比较对应与独立配对，部署仍只由合法RGB/L生成单LoRA。不是已验证修复。
定向官方MimicGen源码核对确认相对目标变换、末端反馈与段内路点时钟的范围，不能把它当成现成的全状态专家。
相对几何1-NN、同视频辅助、R2、SEOD/GOMQ和phase的近邻边界见§36–37。
当前仅登记train34/38各demo0/1与init40–43的最多16次迁移工程，预期60–100分钟、上限120分钟、最多0.5GPUh；
四参考及双参考共同初态覆盖不足则关闭固定构造，不换源/边界或追加尝试。函数类与两臂数据边际固定，
可逆的两视频互换不能称独立；新MT使用同数据、既有强参照继续保留。正式数据/Writer/held均未授权。
这里是事前假设和方法决策，不是新环境或模型结果；实际执行状态看progress。

## 186. 教学迁移有真实跨初态成功；原始scene核对否决了部分配对声明（2026-09-27）

执行29c35f26，train34/38固定四源×四init的16次新轨迹共14成功/6,079控制步，渲染147秒/.040833GPUh，原件156MiB。
主讨论独立读取全部NPZ确认真实动作、T/T+1几何、RGB时间和官方成功首次终止；源只恢复保存state/XML，未推进。
34的四对具有相同保存起点且两参考均成功，证明本有限构造确能从不同物理初态执行同一教学参考。
38同state两参考的stove初位差7.7–21.5mm，故不能验收“两个task共同物理初态覆盖达标”。
14条单轨迹成功和两个pot2 On从未满足的失败原样保留；不把初始化违约当作学习假设阴性或控制器修补理由。
设计§7仅授权一次完整scene配对修正及task38全部8行，原34和四源复用；45分钟/.15GPUh封顶，不只重试失败行。

同一问题还触发对当前四个条件速度面板的1,600份原始t0字段读取：V270/L270与V270/V450各400行body位置、EEF、夹爪一致；
L450在global39（libero_10 local9）的24行microwave位置不一致，最大21.47mm。其state IDs为
`[3,5,7,9,11,13,15,17,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47]`。
这些行在V270/L270/V450/L450均失败；V450/L450的84/18/28、L450/L270的89/23/58数值不变，但对应保存初位一致的分母为376。
原151/147→102/112为真实400行边际分数，不能把24行scene差异抹去或继续说所有400行完整物理配对已验收。
强MT/Source旧面板无同类continuous body trace，缺证据不冒充补核通过；历史分数和强参照继续保留。
没有新模型/环境或旧评测重跑，不将此缺口当作条件速度退化的统一根因；V相邻退化仍在400行保存初位一致的比较上成立。

读取范围只限现存body位置/末端姿态/夹爪/谓词，不称恢复了全部模型静态属性；原固定8 full双RGB核对仍保留其小范围。
原件分别为`.codex/tmp/demonstration_transfer_main_review_20260927/raw_review.json`与`formal_initial_position_scope.json`。
34的两参考在同x0给出不同动作且都成功，但首10步为人工连接段，故不能把这种差异当作有益视频动态的正证据；
后续仍需合法RGB获取、真实唯一LoRA学习、未见任务闭环及能力保持，不由数据构造通过自动进入大训练。

后续完整方法判断补充于机制§37.8：对应P相对独立I的理想FM风险差，取决于视频在自身观测/状态、带噪动作、语言之外
能否改变条件目标均值；源参考相同不保证这一增量，静态画面足够或无益风格也可解释拟合改善。
这约束完整P/I和闭环解释，不建立新的局部探针/课程。只读既定36训练task的BDDL得到30项On/In、6项其它操作；
其中推盘仍不适用现有抬升分段，语法兼容数不是可用教师数。四源/两task不足以替代跨任务方法检验，
后继统一构造若支持不足则关闭，不通过逐task控制补丁或多采同task来制造覆盖。该判断未扩大当时8行纠正范围。

修正81dc1e45完成后，主讨论独立读全部新8份NPZ/4份scene：四对保存初始body/EEF/夹爪/谓词和双RGB一致，
T/T+1/限幅/首成功终止成立；7/8成功，失败为38/state43/demo1，保留不重试。完整model/sim/controller起点另由运行前assert及exit0支持。
旧34共同成功4init、新38共同3init，构造的预注册最小覆盖通过；实际两批24次/9,908步、累计.065GPUh，不混写成24条合格训练数据。
源码集成及结构例外见设计§7.3，原件`pairing_repair_acceptance.json`；没有神经模型/LoRA或held收益。
后继另冻设计§8/机器规格：27个预先选定刚性搬运task、4固定源、新init44–47，最多432次/1.5GPUh。
原四源复用而旧查询不混入，避免没有完整sim快照的旧34被当作四源共同场景。任务支持不足则关闭，不新增控制特例。
这是一轮完整数据支持准备；没有把工程通过改写为理论或最终方法通过。

完整学习取舍随后落入设计§9：保留36task及原始跨episode支持，在支持task中固定原始/新查询各半，仍为同一LoRA普通FM。
P/I只改变新query的条件对应；四源×四视频16格避免固定互换的可逆code，整块视频边际相同、实际query及FM随机流逐事件相同。
该混合目标和有限采样不能冒称纯新数据Bayes风险恒等式已预测网络收益；旧/新事件都没有额外deployment输入。
真正需要完整闭环区分的是对应收益、共同数据收益与两者均未解决；相同数据强MT/已有155、未见task与保持要求继续保留。
这不增加当前§8执行范围，也不从示例288更新自动启动神经训练。

## 187. 固定对应查询获得跨任务支持，但尚未验证条件学习（2026-09-27）

完整构造108固定源中100新兼容、4旧复用、101四源XML/BDDL注册不兼容。416实际行为/71,520步，367成功、49物理失败；
16计划行因源结构未执行，未换源或重试失败。四源共同至少2init支持恰20task（Spatial5/Object4/Goal3/Long4/meta90四项），
所得296条等权交叉由主讨论从raw rows重新计算一致。416 NPZ/104 scene的初态、RGB、T/T+1/实际动作/首次成功停止独立核验通过。
详细来源/中途0动作工程修正/完整计费和结构验收见设计§8.6；此次.573611GPUh，低于1.5上限，源码f8dd3770集成4aeaf5b5。

被搬物未变并不等于整场景未变：删除的单物体门槛不是§8成功条件，不能因这一合法工程修正否定全部采集，
也不能将数据宣传为每个物体都具备明显几何变化。该case其它body差17.5669mm，全部case保存body最大差至少4.8903mm，
仍有16个case-物体项位移<=0.1mm。共同init交叉只消除固定组内的参考—初态选择关联；任务/成功支持和参考造成的occupancy限制仍在。

这些新事实使同一示范参考与新初态动作的真实联合数据不再只是设想，足以实施完整方法比较；
没有证明合法视频已提取操作知识，也没有说明共享编译/优化能保持作用。P/I同query而换教学对应才能检验这项信用，
同数据强MT及真实held闭环/换正确视频/相邻保持用于区分数据收益、条件收益与无益风格拟合。
原36任务保留，新支持20项旧/新查询各半：每完整288宏更新为32,256query，新8,960/旧23,296；不把混合风险套作纯新数据Bayes风险差。
原训练配置明确demo0–45；前次设计§9写50条属文档错误，在任何模型运行前更正，46–49保留诊断，held50视频合同未变。
当前只登记完整模型工程/真实profile（设计§10），不以数据通过预授权大训练。是否已派发及实际运行见progress。

派发后主讨论从既存80源/296查询进一步限定机制：只有34/37/38是双物搬运，其余17task单物；每task四源物体次序全部相同。
故不把该P/I比较讲成“解决顺序冲突”。按拟定均匀采样，新FM标签时间格有75.081%来自变换后路点执行，
8.580%连接、16.338%末动作重复；query在共同初态的权重3.342%。这是目标采样权重，不能冒称梯度/收益比例。
后续occupancy仍依赖参考，自己的状态/带噪动作可能已透露c；P/I同query与held闭环才能检验视频是否新增有益作用。
原件`learning_support_semantics.json`及设计§9.3记录范围，不据此改loss/数据或追加局部探针；已派工程合同不变。

## 188. 工程图接通不等于干预路径已执行；两项限定修正（2026-09-27）

44cf63fa六文件+1000行实现公共128+完整视频16的唯一144 LoRA，source冻结、实际普通FM/cotangent/VJP和ECP恢复已核。
P/I共12实际宏步/1,344query，CPU计划新8,960/旧23,296、36task等权及20task的4×4边际成立；三项CPU测试通过。
最长105帧完整28-query为18.594秒，allocated20.746/reserved21.143GiB；全部五GPU段exit0、完整.206436GPUh。
详细验收与原件见设计§10.6及`learning_engineering_acceptance.json`。没有新closed-loop收益结果。

源码与事件流另揭示：所有支持task每轮同步取旧/新（每9宏步新数0/20交替），首4步和profile全部旧query。
CPU NPZ读取证明不了新query已消费，P/I smoke还未真正干预新数据的对应；不能让工程报告的“完整”代替实际链检查。
原机器规格没有写明task相位是主讨论合同不够具体，后继固定升序ordinal奇偶相位，边际和目标不变。
第二条接口global38教学读取前未取消global2生成LoRA；Meta hook是加法，不会移除原投影适配。
因此该case不具备独立冻结source生成资格，但其动作/状态原件仍真实，失败不定位任何科学根因；训练functional_call不受此顺序case问题污染。

原始梯度显示第2步Core/Text/VL及调制活动、Procedure/Action为0，第3步后全活动。FactorHead末层和Procedure调制串联零初始化
给出相符导数解释：先head、再调制、后Procedure。主讨论“第2步起全部组”要求过宽；只更正文义，不做模型/初始化修补。
后继§11限固定相位、新query真实反传/更新和source隔离：P/I各fresh6、P2→6、一个受影响接口，.40GPUh封顶。
不重复有效原工程、不按失败改loss/rank/权重，不把这两项工程修正升级为EMBER的统一根因或已证实修复。

主讨论并行确定完整比较口径（设计§12）：Writer每步112与强MT每步576查询不同，不能共用“.85GPUh/288”估计。
同数据MT每task16、支持task8旧/8新，总旧416/新160；新采样须保持源/init/时点层次权重，不能均匀平铺296轨迹的所有帧。
28-query实测FM线性外推576×288为4.366GPUh，旧真实MT稳态外推为11.893；均非新MT完整profile或正式预算。
已有强155不因新MT或更小查询池降低，P/I优劣不单独证明部署视频必要。物化采用公共128一次+400份完整条件16，
张量约2.101GB而非400份完整144的18.535GB，执行仍为唯一LoRA。新比较必须共享真实canonical scene，旧参照缺同类原件的边界保留。
这改变下一完整投入的预算和对照实现，未增加当前§11模型工作，未产生新的科学收益证据。

## 189. 对应学习工程缺口关闭；完整比较须同时保留强参照和物理配对（2026-09-27）

399e2610已集成为c9e25e95，四项CPU检查通过。主讨论重算288事件的task相位、4×4格和P/I边际，
实测16更新/1,792query=252新+1,540旧。第6步task97中P/I使用相同真实query及flow seed、不同合法教学0/3，
真实完整144 FM cotangent约.00977，并由同一公共/条件参数消费；这关闭“新干预还未进入模型”的工程缺项，不能证明对应学习有益。
五套ECP保存optimizer/scheduler/sampler/rank RNG，P2→6的原前缀与后续事件/LR成立；正常数值差异不追逐。

每次Runtime.compile恢复物理source identity；顺序接口先装global2，再读global38时偏离由1.234047归0。
唯一global38失败的520动作/521状态、104重规划双RGB均核，属于接口事实，不推断历史失败原因或正式能力。
新增完整.236187GPUh、四段exit0，工程累计.442623；closure1.141GiB/全study2.544GiB，代码1774MiB，均在限额内。
独立原件`.codex/tmp/demonstration_transfer_main_review_20260927/learning_closure_acceptance.json`，完整边界见设计§11.3。

下一投入以完整方法比较为单位，见唯一active design§13：M保持36×16、rank128及真实FM，只做3步和1→3恢复以测实成本；
P6/I6/M3的紧凑bank和共享完整scene进入官方队列，固定24条train接口。P/I不追加学习，工程成功数不参与科学判断。
这是执行完整比较的实际第二消费者和接口整合，不建立表示/范数/loss课程，也不把完成工程升级成已证实修复。
正式预算须同时包含忠实MT、视频物化及official400，不能把Writer局部FM吞吐作MT成本；未知成本先以1GPUh封顶验证。
已有强MT155保持参照，新匹配数据MT较低不能降低目标。P/I差额只检验训练参考对应，仍不能替代视频必要性、绝对能力和相邻保持。

主讨论另在设计§14事前固定正式比较框架：以完整288/576对应格控制投入，因果差按同节点解释，
最新点与相邻成功丢失共同裁决，不择孤立峰值。I若更强也保留为经验候选，不为支持P假设选择较弱臂。
先看P/I完整方法是否值得投入，再做忠实同数据MT与必要视频因果证明；M保持自己的强训练/选点而非套Writer预算。
I依旧读取视频，P/I差不能代替语言参照；M的参数化/rank/task batch不同，也不能由P−M唯一定位动态视频。
当前尚缺本次完整实测的正式成本/执行冻结，§14不是GPU授权，不改变执行者正在承接的§13。

## 190. 完整比较链路准入成立，工程事实与方法证据分开（2026-09-27）

38303429已集成481beb22，主讨论八项CPU检查与24份NPZ/PT/8 scene、576事件及M五更新/三套ECP核验通过。
M实际2,880query、支持task8旧+8新按各半权重，完整36task等权真实FM成立；P/I此次没有新增更新。
共同保存起点及六份full首帧RGB成立，完整model/sim/controller依已审首动作前assert；8,665实际动作和T/T+1无缺项。
P/I各1/8、M0/8只作工程原行。完整GPU .41625、study1058MiB，约73分钟含实现；详细来源和限制见设计§13.5。

报告两处更正：日志是shared A/B整组范数，不能说两个因子各自活动量；guard有两处新增复杂度hard信号，
主讨论按单一集中式分派owner登记窄例外，不把REVIEW无hard作为事实，也不因此拆平台/请求Owner审批。
配置envs8而实际singleton shards只测到每replica1，后继预算不使用虚构八倍吞吐。
PT保存重规划前5计划，成功停止可留下末段未执行指令；实际T以内与NPZ动作一致，不将未执行尾部当执行。

下一完整科学投入冻结为P/I fresh288各32,256query与各correct400，预算含全部阶段18GPUh，预计4卡4–6小时。
新对应干预占5/18查询，尚可能被自身状态/带噪动作解释、仅学风格，或无法经共享学习保持；不预认已取得视频知识。
先将已验收工程运行面转换为正式范围并合入main，随后登记精确commit启动；不另跑重复smoke、不同时投入MT长训练。
§14预留576与忠实强MT继续/停止线不变，288不证明相邻保持或动态视频必要。详情及成本依据见设计§15。

机制§37.9补足实际作用位置：零B identity点公共B的FM梯度只取决于相同自身query，
对应干预首先改变共享FactorHead末层的“执行cotangent×216维教学feature”关联，q/v头还跨18层汇总。
离开identity后公共参数和条件表示共同改变，故最终公共β差异不能被另判为视频已学好但底座破坏；
链式导数说明干预位置，不保证有益参考可辨识、相关经求和保留或闭环提升，不触发新局部probe。

## 191. 首个正式节点可执行；恢复收尾修正不改变科学干预（2026-09-27）

CPU转换3bee0a1b已由9b480d2d集成，原source/model/operator/data/optimization/mt字典全部相同。
P/I288、ECP72/144/216/288、validation8×50紧凑bank/共享scene和official capture已进入唯一运行面；
held编译只构造authority/RGB store，不构造FunctionalQueryDataset，旧M3/P6/I6活动入口退出。
独立八项CPU测试通过；guard REVIEW、无本轮新hard，原中央分派例外仍保留。

主讨论限定修正macro288恢复的终点发布：零剩余更新时先经同一ECP owner在新attempt保存完整恢复状态，
再写真实存在的final checkpoint指针；CPU控制流回归通过，不涉及新的模型训练、指标或模型结构选择。
首72之前无ECP的故障仍须报告，不承诺任意失败自动恢复。完整证据与限制见设计§15.6。
当前正式GPU尚待指定执行者现场quota/设备/preflight与启动，工程通过不能提前产生科学收益结论。

## 192. 对应信用有首个完整闭环正例，绝对能力与保持仍未过关（2026-09-28）

首批正式P/I各288、32,256query，P124/I91，净+33/400；R/G/L71/53/20、churn73、Jaccard .49306，breadth均5/8。
Spatial+20/Object+11/Goal−1/Long+3，global3/11贡献27/33，16/23/39均零。主讨论原始800轨迹、400scene、8完整ECP与实际事件验收通过。
完整7.907476GPUh/wall4.331h/峰值2卡/7.083GiB；低于原18上限，不把未花预算转作无限探索。
固定task-cluster bootstrap、官方顺序复算95%[1.25,17.0]pp；不覆盖重训seed或宣称总体显著。
P的RGB断言失败发生在新episode首动作前，23已发布片348行保留；4份未发布片的已完成轨迹重算，保存行为与成败全部相同。
至少额外803保存动作和未知在途未落盘步须区别于正式240,147动作，失败原因未确证。无择优、无修复声明，旧原件完整保留。
独立证据`.codex/tmp/demonstration_transfer_main_review_20260927/formal_stage1_acceptance.json`，细节与原件见设计§16.1。

P/I共享数据边际、自己的query/flow seed和有限训练程序；只改变5/18查询的教学对应。故这次相对收益不能只归因于新增数据量、
label质量或“现在才有真实FM”。旧NativeCorrection/LocalField已经有真cotangent/同参数消费者；新支持关系是本次实际干预。
机制§37.10解释实际零head的信用入口与后续公共/条件参数共同演化，不把公式当观测到的feature语义。
有限闭环正增削弱“对应关系在该完整学习窗口毫无实际作用”的解释；静态线索、更有益公共学习、真正动态操作知识仍不能分离。
I不是语言臂，P−I不能证明视频必要；P124仍弱于旧强MT155，未检验相邻保持。旧MT无共同完整scene，不能虚构严格得失配对。

按既定§14选择唯一预留576第二格；不是看见涨分后新增窗口，也不改模型/数据/学习数值。
CPU续训入口先转换，GPU另冻精确commit，预计8–10/硬11GPUh、2卡4–5.5小时。停止线仍为lost>20或净降>8，
两臂最新均≤155或均不保持即停本组合长训/controls，不转864、不摘首点峰值或冻结公共参数补丁。
此选择推进完整可失败预测；尚不是EMBER方法验收或新统一根因。

## 193. 预留576续训CPU转换通过，科学假设和输入条件不变（2026-09-28）

2c7e66cf已由e655bcfb合入main。九项独立定向CPU检查通过，原六数值字典相同；固定两父288跨freeze迁移，
后续同stage2仍只接同臂最新完整ECP及同冻结身份。真实模型/环境/GPU尚未在新代码执行，不另开smoke。
精确src净+91、全部Python含tests净+143，run619行集中管理当前ECP/训练生命周期；guard REVIEW无新hard。
旧288原件/读取重建由f4a80cd5冻结树保留，当前唯一bank运行面为576。恢复/资源边界见设计§16.5。

Owner进一步追问原B轨迹与规则迁移A的区别。这里改变的是教学参考与监督的来源关系，不是“新场景”本身：
原则上可以使用同一个B初态，但迁移后全条状态/动作轨迹重新产生。尚未直接量化“原B”对“在B完整初态迁移A”的差异，
不能以假设中的大路径/顺序冲突解释旧训练失败。P/I都使用相同扩增数据，124/91不能识别扩增对原数据训练的净收益。
本构造确实依赖训练期特权位姿/动作、人工分段与仿真；部署合法RGB/L的信息墙成立，不会抹去这项训练数据假设。
保持这些证据边界继续一次预留完整节点，不因当前正例宣称动态规则已被学会或配对是普遍必要条件。

## 194. 对应信用的实际每帧/调制作用与正式初始化开启边界（2026-09-28）

Owner要求将长期科研原则写入项目AGENTS，已由0144cf62新增§12并推送；它不记录动态实验年表或增设审批。
主讨论随后只读当前冻结一致源码及stage1前四行metrics，完成机制§38；没有模型、环境、新数据、GPU或参数扫描。

Core的帧选择梯度由最终功能cotangent与该帧Value相对选中均值的作用差决定；初始0.05门不等于整个Value通路只保留5%，
直接Value信用按混合权重w_t分配，且仍须等待下游零head开启。
Procedure的中心化Value读出等于`sum_t(alpha_t-1/T)V P_t`，每帧信用同时含有符号内容项和基于Value功能差的Key选择项；
Key保留未中心化P，不能称整体平移不变或已提纯运动。Core—过程调制的实际梯度分别为`(c*g_f)r^T`与`g_f r^T`，
它解释何种教学/执行关联能进入这个接口，不证明这些特征已经拥有对象/接触语义。详见§38.1–38.3的完整假设和偏导。

正式P/I第一步只有common/FactorHeads活动，第二步Core/Text/VL/Compiler活动，第三步Procedure/Action Meta活动；
与零head→零调制→零H/E读出output的最早开启顺序相符。首步common组同0.037758935、head组0.066910774/0.067108624；
前四步total norm都<1，clip未缩放。组记录不能证明各内部读取参数何时开启，也不能由小梯度宣称学习失败。
最早P/I分离不需要顺序路径，后续过程活动不识别最终收益内容；这不是新的初始化bug或旧下降根因，不触发修补或S0重跑。

真实执行Q读取自身prefix/suffix，V-LoRA只写suffix；50位置FM经10步flow、执行前5及闭环重规划形成行为，
因此参数/特征变化不等于有益控制。原B对迁移A的直接比较仍缺；P/I不回答构造数据相对原数据的价值。
回查原始7a525a32保留理论竞争解释，也承认未落实该具体比较；不由当前正例事后补称它不必要。
本节收紧完整方法解释，不提出新的架构、probe或GPU任务；576仍按原资格、预算与停止线裁决。

## 195. 对应学习576不能保持首点优势，按事前规则关闭完整组合（2026-09-28）

主讨论独立核bc729e86正式原件：P/I各288→576，新增32,256 query，父数值/拓扑、metrics前缀、第二4×4格、
8套完整ECP与rank RNG成立；800 NPZ/PT和400scene配对通过，238,519真实控制步。没有模型/环境重跑或held标签读取。
P108/I107，同点R/G/L70/38/37、churn75、Jaccard .48276；P对288为74/34/50、净−16，I为66/41/25、净+16。
P/I breadth6/5，Spatial/Object/Goal/Long为35/29/26/18与18/34/36/19；global23/39仍均0。
任务bootstrap[-10.25,+12.5]pp仅八任务单seed描述。旧MT155只有历史标量，没有同scene的严格配对原件。

两臂都丢>20且最新都≤155，原设计§14两条关闭条件均满足：不接864、强M、controls或rank/LR/seed/aux/公共冻结补丁。
首288的P124/I91有限正例保留，但不再支持本完整组合能稳定获得并保持有益条件能力。科学阴性不改写为工程bug。

成功集合进一步限制解释：原P独有53行，576后16仍P独有、11两者成功、7转I独有、19两者失败。
因此总差33→1同时包含I追上和P自身丢失，不能只说I追平；新P/I仍38/37不同成功，不能说收敛成同一策略。
global3差值16→19保留，global11的+11→−9和global26的−1→−10等重分配抵消它；不是所有操作统一失去对应作用。
两臂后72宏步在新/旧查询上的训练FM均比首节点前72步更低，但它们是不同学习时点/样本的训练均值，
不是固定风险或闭环代理；不由它定位encoder、公共LoRA或优化器根因。具体数值在独立验收JSON。

完整GPU7.724252<11，含全部六段加载/执行，全部exit0；两正式批次共15.631728GPUh，峰值2卡。
新wall3h56m49秒、study14.12GiB；所有原件保留。设计§17.3仅授权必要CPU退役，不恢复其它关闭实验。
原B轨迹与迁移A标签的直接差异、合法视频是否提供有益动态知识仍未知；这限制可推广结论，不能据未知无限保留当前主假设。

## 196. 迁移参考含实质相对目标差，但额外控制价值仍未被识别（2026-09-28）

机制§39将当前实际规则写为：源相对目标xi/Omega，经query在segment进入时的参考body变换为OSC目标，
再用自身末端误差、.05米/.5弧度尺度和限幅得到动作；每段另有10步连接，参考时钟并未被状态反馈替代。
这解释了新初态动作如何得到，也明确其不是任意自身状态上的完整反馈专家。

主讨论只读既存train34/38各demo0/1四条source，比较旧四段各自首末OSC目标在参考body坐标下的差异：
16个边界位置差17.30–49.23mm；末点旋转差最高32.87°。初位/朝向消除后差异仍在，不能把两源视为仅世界平移。
34最后一段末的源夹爪命令为−1/+1，相同任务不要求保存末动作相同。分段来自旧启发式，未认证相同接触阶段；
目标差不是实际动作差、闭环优劣，也没有补齐原B对A→B的同完整初态轨迹比较。
原始sources根沿旧demonstration_transfer_engineering_20260927，读回JSON/脚本在主讨论tmp的`reference_operator_readback.*`。

新增方法约束：有益参考差异应与自己的误差/参考朝向形成可用交互，而当前E/H/Core/Procedure没有被实测为该几何表示。
共同identity点P/I的head信用差只消费参考间delta h；共同h项相消，不等于整个训练的任务语义信用为零。
故P/I是对应干预，不能单独区分有益相对规则与实例路径细节，也不测共有操作知识是否已经取得。

原B与A→B比较还改变query状态/像素和FM插值输入；局部`delta r=[I+(1-tau)J_z v]delta a`说明标签差不是固定输入上的全部信用差。
即使B→B也经过连接、时钟和限幅，不应默认等于原B。上述缺口现在已具体到生成算子与真实训练消费者；
未实施的对照仍明确未实施，不因未知而补一整套矩阵。当前组合已关闭，新方法需改变完整功能联系，而非沿原Writer添监督补丁。

## 197. 对应学习关闭后的活动源码退役已验收（2026-09-28）

3bad97e6以3c364a4a集成；13文件+294/−3434、活动源码净−2175。私有训练/采集/物化和无消费者配置/测试删除，
封存bank唯一读取owner与官方scene恢复保留，没有fallback。四套正式P/I×288/576使用其原冻结spec/checkpoint/Git，
不因main继续变化失去来源，也不让工程bank混入正式历史；全部原件、数据、checkpoint和冻结树未改。
主讨论独立pytest4/4、四套official capture合同与共享导入通过，guard REVIEW无hard，详见设计§17.4。
未运行模型/环境/GPU或重评测。退役是已作科学裁决的工程收尾，不是新的性能、根因或方法收益；研究目标仍未完成。

## 198. 阶段内负反馈不辨识教学目标，原B比较须分开规则自身效应（2026-09-28）

机制§40对真实迁移规则作有条件推导：固定路点/未限幅位置下，末端与参考平移的动作导数为−D^-1/+D^-1，
不同教学共享；参考右旋转的导数为−D^-1 R[xi]_x，才保留参考目标。限幅、夹爪sign与阶段切换另有条件。
所以只增加末端状态响应监督可能加强通用控制，却没有辨识视频目标；值监督、状态反馈和阶段选择不可合并为一个“控制语义”。
native attention对自身观测变化的Value项和选择项已明确，真实消费者还含十步flow及五步环境动力学。
未测得的特征含义/控制稳定性/跨checkpoint保持不由这些导数推出，也不据此解释576唯一根因。

已对照旧物理J、语义rank、LocalField、phase learner-state aggregation和条件速度，不恢复这些局部探针或原Writer稳定性补丁。
本轮只作历史/源码/指定原始方法读取；新的实证尚未执行。
下一项`reference_transfer_comparison_design.md`固定4个原B起点×R/S/X=12条：原动作重放、自身参考、另一参考。
原保存B另只读比较；源未保存的controller历史明确未知。该设计分开规则自身与参考差异，能约束未来数据取舍，
不把差异大或controller成功当Writer/视频收益，不重开P/I或原数据/P/I/M训练矩阵。

## 199. 同场景R/S/X区分了参考、时钟与重放误差，未验证新标签优于原B（2026-09-28）

固定train34/38×demo0/1，b1541155完整12条/4548动作，主讨论独立原件核验通过；main集成7bb1d558后退役唯一runner。
四组R/S/X成败依次0/0/1、1/1/0、1/1/1、0/0/1，共2/4、2/4、3/4；没有Writer、模型训练、held或追加样本。
共同scene/首帧双RGB、T/T+1、R原HDF5动作、S/X实际目标变换/源时钟/夹爪及首次成功停止成立。
原件root `/data1/user/ymdai/ember_runs/reference_transfer_comparison_20260928`，完整113秒=.031389GPUh<.20、峰1卡。

R对保存B的EEF p95差4.59/5.15/1.99/86.76mm，谓词不一致159/1/0/14状态。历史controller/夹爪内部值未保存，
不能称原B精确重现。保存observation EEF与控制器forward后site pose又是不同接口，不能据前者精确反算逐步动作。
这限制推论，不把正常采样差异变成当前Writer根因或新的重放修补任务。

原时钟R–S EEF中位差93.7/102.4/74.1/73.8mm；S均多40连接步。主讨论事后仅按记录source_step对齐同参考的原动作，
排除连接后，动作后EEF中位/p95为2.65/5.22、2.55/6.31、.96/2.21、12.27/86.03mm；对应谓词差4/0/0/253。
没有DTW、A/B自动相位匹配或结果选点；前三例的大同时刻差不等于大操作差，第四例仍有真实路径/子目标差。
参考更换影响当前行为，但无一致有益性；四固定案例不估计总体，不将X3/4当更强教师。

完整裁决见机制§41：进一步降低“逐示范参考对应+旧Writer即可解决”的支持度；不把原跨episode B先验称为不相容。
P288正例和576阴性均保留，当前比较不解释576唯一根因，也不证明合法视频已经编码出所需关系。
不补数据/P/I/M矩阵、扩构造、修控制器、筛视频或恢复已关闭方法；本数据比较线完成并收束。
独立CPU脚本/验收JSON在`.codex/tmp/demonstration_transfer_main_review_20260927/reference_comparison_acceptance.*`。

## 200. 同一实际A的教学写入/执行读取形成新候选，尚未验证（2026-09-28）

机制§42重新核LocalField源码50dafb05：它已经有逐位置UrX^T和真标签/最终FM共同消费者，不能以缺少绑定或同一梯度为由重提。
本候选的实质假设是公共完整rank128的A，同时作为教学key投影与实际执行读取；公共β还参与合法教学native读取，
逐帧变化Value写入的矩阵M直接成为最终B0+M。Writer一次前向后固定唯一38-target LoRA，不运行部署优化器或在线视频记忆。
具体图/梯度/对照/资源见`docs/designs/operator_read_write_learning_design.md`，不是已验证根因或修复。

新增解释把第t个视频Value的真实信用写为`delta * <A h_query, P_t^T k_tj>/50`，P_t是后续矩阵写入的传播；
A还同时接受执行读取和公共β改变teacher表示的信用。对照只将教学key改为独立S，保留相同β/V/M/真实FM与初值。
公共A可能学得更合适的匹配，也可能受共同空间限制；LoRA坐标自由度、state-free/自身state差异、末帧覆盖和跨更新干扰均保留。
递归非扩张界不证明动作/闭环/相邻保持，constant rank128可达也不保证学到；历史ProcessPullback、LocalField、条件速度与P/I失败继续约束投入。

选择有界工程而非新长训：T/U fresh4和一次T2→4，1120真实query；长视频28-query无更新；唯一train-only canonical病例。
预计75–120分钟、完整GPU硬限1h/峰2卡、原件6GiB，新内容全data1；具体派发以progress为准。
不使用迁移数据，不补原B/P/I/M矩阵，不运行held/controls或恢复旧方法。正式预算和完整方法选择待工程成本及后续科学合同。


## 201. 同算子读写工程路径成立，尚无共享约束或视频收益证据（2026-09-28）

9766cb8b完成T/U fresh4和T2→4，共10实际更新/1120query；原件为
`/data1/user/ymdai/ember_runs/operator_read_write_engineering_20260928`。
主讨论独立源码、五ECP/metrics/事件与唯一520步病例T/T+1/PT核验，CPU3/3通过，完整0.3334568577GPUh。
两次零更新失败仅修梯度记录器，不改变算子/数据/损失；失败尝试与成本完整保留。
真实公共β在教学native前向运行，最终唯一A/(B0+M)经FM/VJP回放；第一步O/B0活动、第二步起A/P/C/D/native信用和U独立S活动。
这不隔离公共β教学梯度的相对贡献，不证明学习到控制语义或跨初态泛化；唯一病例失败不能裁决完整方法。

单位化教学key与未单位化执行read给A不同信用：前者局部只改变投影方向，后者还影响幅度；同参数不等于梯度自然一致。
完整T/U干预检验这项耦合的有限学习收益，不能唯一归因于语义；同β路径在两臂保留，也不由T/U单独识别。
下一步§8预注册各fresh270的完整节点及固定MT同scene参照，只有接近强参照才进入预留450，且须检验成功保持。
当前只派CPU转换，未启动正式学习/held/controls；工程通过不是自动科学晋级，也不摘旧候选峰值或重跑旧实验。


§201后续CPU事实补充：01242f6c保留model/native和source/operator/optimization原数值，完整36task事件与T/U/固定MT同scene运行面已验收。
主讨论8+4项CPU检查通过；旧MT原训练/评测Git normalizer与当前相同，旧临时路径消失不等于数值不同。
新MT比较仍使用原step300和唯一共享rank128，旧155只保留标量；当前没有本候选正式性能/共享约束收益或视频必要性证据。
精确计算阶段和预算以设计§10/progress为准；结构例外与源码生命周期见设计§9。

## 202. 输出rank与跨任务共享子空间须分开；当前不限制Writer参数预算（2026-09-28）

Owner澄清MT最初rank128的预算匹配目的，随后明确当前以性能为先、Writer参数量不设限；不是要求恢复参数匹配。
主讨论撤回“MT为128所以Writer也应128”的选型推论，也不将此次提问转为新容量约束或rank实验。
实际T/U可训练参数42,352,640/47,206,400，MT为10,297,344；输出rank和学习参数总量是不同口径。
当前每个target的DeltaW_c=(B0+M_c)A共用A，因此stack_c(DeltaW_c)的秩至多r；
逐条件生成A_c的旧rank16只限制每个DeltaW_c的秩，不要求跨task联合行空间仍至多16。
这说明直接把新图r改16不等于恢复旧rank16函数类，但联合权重表示条件不等于任务成功所需的行为容量，
不能由此断言实际必须128。当前没有16不足或128最优的因果证据，固定rank的T/U结果也不会提供这项证据。

## 203. 同算子首批绝对能力不足；更长学习与绑定收益仍须分别判断（2026-09-28）

原件root=`/data1/user/ymdai/ember_runs/operator_read_write_learning_20260928/stage1`。
新主讨论直接读取三组1200 JSON行与contracts/completion、全270训练metrics，复算成功集合/配对统计和8-task bootstrap，
确认T116/U128/同scene MT153，各400、breadth5/6/6；task/video/scene/seed和policy-noise前缀配对。
实际PT/NPZ全内容检查继承执行者已运行的`analysis/verify_official_raw.py`及raw_integrity/orphan报告，未再次扫描1200份capture；
此前actual270事件、ECP元数据、bank来源及839窄修验收继承，不重训或重复GPU。

| task | T | U | MT |
| --- | ---: | ---: | ---: |
| 3 | 9 | 19 | 41 |
| 6 | 9 | 5 | 7 |
| 11 | 31 | 41 | 36 |
| 16 | 0 | 2 | 10 |
| 23 | 0 | 0 | 0 |
| 26 | 35 | 37 | 35 |
| 31 | 32 | 24 | 24 |
| 39 | 0 | 0 | 0 |

T−U R/G/L/churn=87/29/41/70，T−MT=90/26/63/89，U−MT=95/33/58/91；
任务cluster bootstrap10000、seed2026092810，95%差额区间依次[-11,5]、[-26.5,4]、[-18.5,2.75]pp。
区间跨0不是等效证据，也不含重新训练seed不确定性；跨臂“失”不是时间上的遗忘。
T对MT净差37中task3贡献32，U净差25中该task贡献22；不能将其余任务的得失抵消误说为全面相同。
两臂在task11/26/31等有实际成功，不能因未达参照抹除；也不能用单task长处宣称整体视频学习成立。

两臂最后两个45-update窗口在线FM仍下降，T .108655→.106422、U .107977→.105724，各22/36 task下降；
不同采样窗口不是固定risk曲线。无nonfinite/触发clip，真实参数与native信用活动只证明学习在执行，不定位功能根因。
270只有30,240 query、每task30访问、warmup后120更新；MT step300为172,800 query，是强能力参照而非曝光匹配因果臂。
因此目前可以裁决绝对能力不足、绑定优越性无正证据、原140投入线未过；不能裁决充分收敛、读写总体不可能或视频必要性。
历史条件速度V/L与P/I后期丢失能力仍约束“多训必好”的解释，不将本次续训当必然修复。

T/U都保留视频/β/M且U参数更多，完整对照只改变key与实际A的共享约束。共享A不能单独证明语义度量或梯度相容。
Owner明确支持有界更长学习，新的§11在不改模型/优化程序下观察450/900与270的绝对能力、覆盖和成功保持；
原预注册事实保留。900后不自动接更长训练、改架构或补controls；完整视频因果资格尚未成立。

旧批完整报告成本11.1582785066GPUh（train7.0628704835、bank .6236525976、official3.4717554255），
包括两次失败及T332行复用后恢复；终值22631660156bytes、峰3卡是已完成记录，连续storage/CUDA高水位未测。
每卡2workers只是当时模板，后继必须按现场资源/吞吐选择，不能冒称Owner或硬件限制。

## 204. action_in固定key有精确读写覆盖与时间传播，但未定位性能根因（2026-09-28）

机制§43从实际`operator_writer/model.py/native.py`推导action_in的固定K递归：
`M_n=sum_t V_t(K^T/50)(I−KK^T/50)^(n−1−t)`，自身作用为`M_n A x_query`。
视频经H变化写入V，故固定probe不能推出此层视频无效；这层也不能冒称按帧内容选择语义地址。
只CPU读取既有T/U 90/180/270的相关A/S/probe，没有模型/视频/标签/GPUforward。

实际probe秩32，A/S也秩32，T有span(K)=col(A)的确定覆盖；U270的A能量99.9863%仍落在col(S)，
因此没有此层严重读取子空间脱离证据。子空间覆盖不保证语义、尺度或完整功能；固定U执行A替换寻址S→A，
lag0/25/50的寻址/读取系数矩阵差为4.26%/4.44%/5.46%，未运行Value随key改变的完整消融。

T270传播模态半衰期约5.90/34.12/666.10次写入（min/下中位/max），U270约5.79/34.30/673.64；
原实际训练视频的写入数中位28、范围13–104。传播会加权较早写入和反向信用，完全静止尾段V=0时仍收缩M；
但模态半衰期/传播RMS不是有用信息保留率，未识别真实H/Value的控制语义，也未解释116/128差额。
六节点谱近似保留，不能据此说学习无效。当前降低输入层固定probe无效/严重脱离的解释优先级，不改key/添加保持补丁；
继续有界完整学习，同时保留真实Value与自身功能联系尚未知的边界。

## 205. 旧270的task3差额主要出现在搬动目标之前，内部根因尚未定位（2026-09-28）

机制§45按最大task差额定向读取旧T/U/MT的task3全50场景位置原件，另核固定init0的三份完整PT与合法teacher24 RGB。
没有新模型、rollout、梯度或GPU；这是后验行为描述，不是独立机制实验。teacher24画面所搬为饼干盒上的正确目标碗。
init0中MT搬动目标并103步成功；T目标碗全程不动，却搬起柜子上的另一只碗至盘附近；U未移动任一碗。

全50行中，T/U/MT成功9/19/41；目标碗全程位移<1cm为39/28/2，其中另一碗升高≥3cm为18/8/0。
在MT成功而Writer失败的真正配对集合，T33行中31行目标未动，U23行中20行目标未动；反向收益两臂各1行。
故“只差最终摆放精度”不能解释这个task的主要配对损失，实际不足更早地表现为目标关系未落实到接近/操作。
位置阈值不等于抓取/语义检测，一个明确搬动非目标碗的案例也不证明全部底层控制已经学成。

公共β、视频H/Value、自身地址读取、有限曝光与闭环分布尚未因果分开；强MT曝光更多，不能据差额宣布架构不可能。
这份行为观察和§12训练侧功能诊断不能拼成已定位某层的因果链。继续450/900完整学习及成功保持判断，
沿现有capture观察该缺口是否收窄，不增加标签/loss、held梯度、挑视频或局部架构补丁。

## 206. 原270条件残差有不均匀的真实FM作用；Q8信用衰减差异还不是根因（2026-09-28）

设计§12固定4训练task×2teacher×2臂的16条件已经执行，0更新/0环境episode/无held标签，.0497363GPUh计入原预算。
主讨论直接分析16份PT及真实FM消费者，完整见机制§46。完整LoRA相对其共同训练公共β在13/16条件降低FM，
T/U平均差−.015228/−.016587，主要由task0约−.05贡献；task20的T两条件小幅升高。
故“条件残差在自身query完全无实际功能”不成立，但β是coadapted参照，不是独立language baseline或视频必要性证明。

精确输出误差恒等式`ΔL=2 mean((v_β−y)δ)+mean(δ²)`把幅度和纠错方向分开。
task0预测改动与原误差内积明显为负；task20的T近乎正交/纠错很弱，不能用“幅度非零”代替功能收益。
实际action_out的M直接项RMS仅占完整预测差的4.50%–27.74%，中间hidden变化确实参与；比值不是因果贡献率。
没有逐query FM目标/损失原件，不声称每个自身状态受益；没有因此增加forward或重建随机目标。

同task两teacher原件的CPU分解进一步限制“主要是换视频波动”的解释：T/task20扣除两预测间方差后，
共有预测仍比β增加FM .00036575；task32的U虽波动略大，共有纠错仍强于T。共有/差异分量只是精确均方恒等式，
不是可部署的平均LoRA、独立language能力或视频因果收益，也不授权一致性loss或挑视频。

按实际K、非交换R积与真实FM G_B计算较早Value信用：Q8的首写传播范数比中位T .03572、U .20459，
最后三写的绝对缩放信用占比中位T .9234、U .1329；T同层地址更集中，但V8/action_out没有统一相同模式。
这些量描述当前函数的局部偏导，不是语义保留或帧的因果价值。task0构成实际反例：T尾部信用93.80%/95.12%，
U3.00%/5.44%，两臂仍获得接近的完整FM改善；不能由“尾部集中”直接推出性能根因或启动保持/寻址补丁。

当前认识推进到“已有局部实际纠错，收益不均；目标关系到自身闭环的可靠迁移仍不足且未定位最早内部失效处”。
原450/900继续；闭环、保持与视频必要性仍分开判断，不按此小面板选checkpoint/视频、调scale或增loss。
900后的方法取舍结合完整结果与这些竞争解释，不用未知根因无限保护当前假设，也不由单个局部指标整体换架构。

## 207. 450有真实学习收益，U接近但未胜MT，T的能力置换仍明显（2026-09-28）

Owner指出450已就绪，主讨论在900整批回报前独立消费完成的两组400。原件root为
`/data1/user/ymdai/ember_runs/operator_read_write_learning_20260928/continuation900`，
入口是`{T,U}/evaluation/450/correct400/{results,run_contract,launcher_completion}.json`及`{T,U}/banks/450/manifest.json`。
450 ECP manifest的next_macro=450、world2；训练/评测均来自已登记81846ed3 clean detached冻结树。
Git记录authority_contains_commit=false是原分支commit经cherry-pick集成的身份边界，本合同明确允许已push执行分支，
不据此要求重训/重评测；main的旧MT capture路径窄修不改变T/U450计算。

主讨论直接读取新800及比较所需旧JSON行，复算全部结果；task/state覆盖完整，scene/语言/video/环境及policy RNG前缀配对成立，
各task50视频无放回。`pi05_eval_results.paired_success_comparison`负责得失，operator的scene/video字段另作显式配对。
两个launcher均一次完成、各单卡3 workers全部exit0；8full+392compact、完整NPZ初态/动作/谓词与PT/RGB检查继承
Sol实际执行的`analysis/verify_official_raw.py`及`raw_integrity_450.json`，主讨论已读对应消费者，没有再扫全部800份。
两个450面板报告无failure/orphan trace；launcher56.59/53.76分钟不是整批完整GPU账，900与总成本未在本轮验收。

| 臂 | 270/400 | 450/400 | 450 breadth | 270成功保留 | 新增 | 丢失 | 丢失占旧成功 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T | 116 | 122 | 7 | 79 | 43 | 37 | 31.90% |
| U | 128 | 143 | 6 | 106 | 37 | 22 | 17.19% |

同scene强MT153/400、breadth6。T/U450分别30.50%/35.75%，MT38.25%；U更接近但未达到超过MT目标。
相邻churn为T80/U59、Jaccard .4969/.6424；U保留82.8%旧成功、得37失22，是阶段性保持较好的净进步。
Owner质疑后明确区分：22不是稳定性否决线，也不要求零交换；尚未确认的是多节点持续保持，而非这一节点不合格。
T breadth由5升7，新增task16/23均只有1/50成功，保留为有限正例，不能掩盖原强项退化。

| task | T270 | T450 | U270 | U450 | MT | 任务要点 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 3 | 9 | 24 | 19 | 31 | 41 | 饼干盒上的碗放到盘子 |
| 6 | 9 | 2 | 5 | 2 | 7 | 饼干盒旁的碗放到盘子 |
| 11 | 31 | 42 | 41 | 40 | 36 | 奶酪放进篮子 |
| 16 | 0 | 1 | 2 | 2 | 10 | 黄油放进篮子 |
| 23 | 0 | 1 | 0 | 0 | 0 | 开上层抽屉并放碗 |
| 26 | 35 | 38 | 37 | 41 | 35 | 奶酪放进碗 |
| 31 | 32 | 14 | 24 | 27 | 24 | 奶酪与黄油都放进篮子 |
| 39 | 0 | 0 | 0 | 0 | 0 | 杯子放进微波炉并关门 |

各suite分母100，按Spatial/Object/Goal/Long：T270为18/31/35/32，T450为26/43/39/14；
U270为24/43/37/24，U450为33/42/41/27；MT为48/46/35/24。T净增6包含task3+15、task11+11，
同时task31−18和task6−7；U净增15主要来自task3+12，不能把所有task概括为一致改善。

| 比较（候选−参照） | R/G/L | churn | 差额pp | 8-task bootstrap95% pp |
| --- | --- | ---: | ---: | --- |
| T450−U450 | 106/16/37 | 53 | −5.25 | [−12.25,.50] |
| T450−MT | 99/23/54 | 77 | −7.75 | [−18.00,2.00] |
| U450−MT | 107/36/46 | 82 | −2.50 | [−10.25,4.75] |
| T450−T270 | 79/43/37 | 80 | +1.50 | [−12.25,14.25] |
| U450−U270 | 106/37/22 | 59 | +3.75 | [−1.50,10.50] |

主讨论按原seed2026092810、10000次task重采样复算一致；这些区间只有8个task，未包括重新训练的不确定性。
U与MT总数差10，但仍有36得46失，不等于复现MT能力；这里的跨臂损失与上面的自身训练遗忘分开。
同点差额及自身保持均倾向U，降低当前绑定约束的投入支持度，但不是已证明所有共享坐标有害或U参数更多必胜。

先前§205预定的task3行为问题已用新100份位置NPZ接续，旧150份汇总直接继承。
目标碗全程位移<1cm：T39→22、U28→16；其中非目标碗升高≥3cm：T18→6、U8→1。
task3的T18个新增成功中16个、U15个新增成功中13个，来自旧目标碗几乎未动的初态；两臂各丢3旧成功。
固定init0从T搬动另一碗/U两碗未动，变成T143步/U93步成功，450两臂另一碗均未移动。
因此既有实际行为缺口随原训练收窄，不能把270该缺口当作完整架构硬性不可能；仍未隔离公共β与视频映射的贡献。

T的task31原32成功只保留12、另增2、丢20；丢失20行在450末态为仅奶酪7、仅黄油5、两者均未在篮内8。
全50行奶酪谓词曾真而末态假由5增11；这些是环境谓词，未因果定位内部时序表示，也不等于所有失败都发生在第二物体。
task11单奶酪能力上升而task31组合能力下降，两task还存在场景/初态差异，不能当作只改变组合性的严格消融。
机制解释及对§46局部证据的约束见§47；不把270训练Q8信用衰减直接说成450 held Long退化根因。

三项性能目标分别更新：绝对能力有进步但尚未胜MT；correct-only没有新的视频有益增量鉴别；保持方面T有明显局部退化，U阶段性较稳健，持续性待后续节点。
按事前§11完成原900，450不提前选最终模型、不改训练、追加controls或自动接1200。900需检验修复是否持续、旧能力损失是否继续，
而不是只追最高总数。此轮无新模型/环境/GPU/held梯度；CPU汇总在同名tmp的`main_scientific_450.json`、`main_task3_behavior_450.json`。

## 208. 450的T/U差额集中在目标关系与双物任务，绑定功能须在真实450特征上区分（2026-09-28）

进一步CPU读取task31四组各50份谓词轨迹：奶酪曾入篮/黄油曾入篮/奶酪末态丢失为
T270 47/32/5、T450 39/21/11，U270 39/28/3、U450 34/31/1。
U两目标都未曾满足的失败7→12，说明总体保持改善不等于所有状态环节改善。
270两臂都成功的同16个scene/video，450时T保持10、U保持15；该差别不是旧成功集合不同造成的表象。
U450相对T450总差21中，task31贡献13、task3贡献7，其余合计1，限制“统一全模型故障”叙述。
视频内写入覆盖、跨训练更新的能力丢失和闭环中物体离开目标是三个不同事件，M部署固定，不能混称记忆遗忘。
推导、完整行为表和竞争解释见机制§48；当前仍没有唯一内部根因或视频必要性结论。

Owner要求现在深入450；设计§13登记原16个训练条件的450功能读取及冻结U的S:=A干预，
比较共同β/条件纠错与独立地址的当前功能作用，0更新/0环境/无held标签，不改900或选点。
脚本已准备，派发/实际执行和结果只看progress，不将计划或CPU语法通过写成GPU诊断完成。

## 209. 450实际功能与回绑：U独立投影有用，T训练图亦有效，不能据局部负夹角定位闭环根因（2026-09-28）

§13已完成并由主讨论直接消费16份PT/实际脚本/全部rows/合同与退出账；原件
`/data1/user/ymdai/ember_runs/operator_read_write_learning_20260928/continuation900/analysis/teacher_to_query_450`。
81846ed3、450 ECP、原固定train0/12/20/32与两teacher/28query/flow均未改；0更新/0环境/无held标签。
外部133.4325秒/.037064589GPUh，exit0，16PT共820,253,192bytes；该成本计入原30GPUh，未验900整批总账。

450的完整−共同β FM均值T−.017407/U−.017309，全部16条件有益（270为13/16）；
完整FM均值T .121575→.116190、U .120018→.116039。T原task20两条无益残差均转为有益，
平均残差收益增加也比U大，不能用“U训练视频分支普遍更好”解释held143对122。
公共β与条件残差共同变化；上述记账不是独立因果份额，β也不是独立语言baseline。

U固定其它参数与native、S:=自身A并重算K依赖Value/M，8/8条件FM上升、均值+.010440，
其中3个比共同β差，支持当前独立投影有实际功能。它不等于重训T、纯寻址几何或视频必要性。
参数一阶cotangent在4个条件预测改善而实际全部损伤；task0/teacher40实际+.012896可精确分为
prediction变化平方+.013820与输出误差交叉−.000925，限制用一阶符号预测有限结构干预。

主讨论CPU从实际450 native/Value权重/G_B对key求导、与独立最终LoRA G_A并列，48个M重构最大误差9.79e-7。
T的Q8/V8/out负内积分别4/8、8/8、6/8；但两路合成后的局部一阶贡献在23/24个组合均降低误差。
V8合成平方范数/两路分开平方范数之和.816–.984，不是接近完全抵消；中间β→native、全任务汇总与Adam未由此覆盖。
故支持存在局部牵制，但不支持宣布严重冲突就是T450退化根因。action_out的无间接路例外见机制§49.4。

T Q8地址λmax约.7506→.7533、早期传播比相近，末三次绝对Value信用中位占比却.9234→.5377；
task20/teacher38该占比反增.4484→.8480而残差从无益变有益。局部时间信用可随学习变化，也不单调决定功能。
完整推导、逐条件反例与数据界限见机制§49。当前保留U的闭环/保持优先性、保留T有效学习正例，
后继判断聚焦条件作用的跨状态/跨任务适用与保持；不将此措辞当根因，不新增诊断矩阵、loss、controls或自动1200。

## 210. 900完整结果：T后段改善、U局部退化，绑定解释与后继先讨论（2026-09-28）

原件`/data1/user/ymdai/ember_runs/operator_read_write_learning_20260928/continuation900`。
主讨论直接复算新900共800结果行、既有各点配对行、全部12组成功集合及8-task bootstrap，读取900 ECP/bank/合同/退出，
与执行者报告一致；完整训练/ECP和新1600 PT/NPZ检查继承已审实际消费者与验收，不重复全扫。
新增task3/task31共200份900小型NPZ用于同一行为问题，900训练metrics用于学习轨迹描述，无新模型/环境/GPU。
12份外部退出原件均exit0，新增22.169122590GPUh；历史11.158278507单列，不把剩余预算当追加许可。

| 单checkpoint | 270 | 450 | 900 | 900 breadth |
| --- | ---: | ---: | ---: | ---: |
| T | 116 | 122 | 148 | 6 |
| U | 128 | 143 | 132 | 6 |

同scene MT153不变。T900−T450=+26，R/G/L105/43/17，旧成功保持86.1%、task-bootstrap95%[−1.25,16.50]pp；
U900−U450=−11，111/21/32，保持77.6%、[−10.50,3.00]pp。T900−MT为−5，115/33/38、[−5.50,3.00]pp；
T900−U900为+16，111/37/21、[−2.50,11.25]pp。区间只覆盖8task重采样，不覆盖重训不确定性，跨零不表示能力等价。
T的后段保持与净增应正面承认，不能仅因失17例判不稳定；同时270→900仍丢32个旧成功、得64个，不能写成整体单调保持。
T breadth7→6仅由task23的1→0造成，不夸大为大范围覆盖崩溃；task23/39在900均0，弱任务覆盖仍不足。

T450→900净增26恰由task3+19、task31+7贡献，其余合计0；T270→900净增32而task3独增34，其余合计−2。
task3 T9→24→43，450的24成功全部保留；目标碗位移<1cm的行39→22→4，伴随另一碗升高≥3cm的行18→6→0。
U同任务19→31→31，但目标几乎不动28→16→4，移动后失败3→3→15；450→900得8失8。
因此U的总分平台掩盖了行为变化；T/U900的目标几乎未动/错碗升高两项统计相同，仍不足以解释43对31，不把物体移动当已正确抓取或语义理解。

task31双物入篮T32→14→21、U24→27→14。T后段得12失5，奶酪曾入篮后末态丢失11降5；
U得3失16，奶酪/黄油曾入篮分别34→24/31→25，两者皆未入篮12→15；16个丢失场景中8只曾奶酪入篮、
3只曾黄油入篮、5两个都没达成，不能概括成唯一第二阶段或目标保持故障。
270两臂共同成功的16个scene/video到450是T10/U15，到900是T8/U7；T总体恢复并未恢复这一固定共同能力。

每90更新含各task10次条件的在线FM均值，361–450到811–900为T .101624→.095450、U .101367→.094952；
U训练误差稍低而held后段退化，不能用平均训练拟合质量排序闭环，亦不足以确诊过拟合。
末两90步窗已很接近，但query/flow与模型均变化、LR仍衰减，不能宣布已收敛或保证再训有益。
450各task已见50个不同teacher，900是同一50视频第二轮；后段T改善不是增加新task或新teacher种类的效应。

450回绑实验只证明冻结U的已学配合依赖其S，不能排序分别训练的T/U；900反转也没有测出共享A已形成语义度量。
现有270/450训练面板支持视频条件残差参与有效纠错，仍无900对应功能分解或held视频必要性证据；
共同β、条件写入M与自身读取A的共同变化尚未因果分离。完整解释、反例和读过/继承范围见机制§50。
Owner最新要求分析后先讨论：本轮不最终选点、不宣布路线通过/退役、不派Sol、不启动后继或补控制。

## 211. 冻结900已有内容匹配的功能，尚不能归因为可迁移过程规则（2026-09-29）

设计§14A/C已完成，原件分别在`operator_chain_diagnosis_20260929/train_functional_900`和`train_nearfoil_900`。
80个功能前向及24个新增近邻条件，0更新/0环境/无held标签；成本分别.099073932和.05182350GPUh。
主讨论直接审脚本、配对、退出和全部104份预测/真实train target，复算实际FM；完整feature/gradient finite/shape
继承已审消费者。源码、固定面板、逐query/前缀/补尾、实际M与函数反事实及计算边界统一见机制§51–53。

固定目标语言与28个自身query/flow，四个训练task的两个正确视频，在T/U共八组内各自优于所有六个错来源视频，
16个正确也各自优于共同训练β和强MT300。近邻补充保持语言/query，六组内两正确又各自优于四近邻；
0/12同操作近邻通常比远来源有益，T20却近邻略差于远来源，不能把同场景相似度当普遍功能排序。
错来源仍有33/48条件改善β；正确优势不只是错误组方差较大，也不排除静态内容门控或训练task模板。
β是共同适配的分支，不是独立训练的语言baseline；这两项不证明时序必要性、held闭环视频收益或最终资格。

同一固定query的完整FM从270→450→900在两臂四task全部改善，均值T .121575→.116190→.111222，
U .120018→.116039→.110170；450→900共同β反而三task略退化。不能说公共β没有作用，它也生成教学native特征。
固定自身action_out hidden仅替换末层M，不能重现实际全38项换视频的损伤，说明上游hidden与交互具有实质作用；
不是可加层级归因。全套M有限替换96项实际均有害，而69项局部一阶预测为有益，禁止据单层梯度夹角推导删改有效。
固定target后，近邻/远来源M与预测也不能由同一修正乘一个全局标量解释；逐target标量仍有大残差。
这些证据使“视频通路没有功能”“只是通用幅度开关”“只在末层吐动作模板”的强解释失去支持。

实际不足仍具体存在：train32完整50位置FM胜MT，前5位置却T差.005288/U差.008223；
U正确对错误的前5差仅−.000332。它是实际函数排序反例，但单flow速度并非10步生成动作，不能直接授权改前缀loss。
900训练功能继续改善、U的held总分却下降，限制了“平均训练拟合更好就能预测闭环排序”的解释。
当前可定位到所学条件函数的即时作用/跨task与闭环适用仍不足，尚未确定唯一内部原因，未验证任何应重训的修正。
理论说明M对自身Aq的实际读出具有带符号、非交换的后续覆盖，且条件修正共享A/O子空间；T的同一A约束
不保证语义度量/梯度相容/保持，U解绑定也不解除共同执行空间。这些性质须与上述实际功能共同解释，不单凭公式裁决。

视频诊断矩阵到此收束。Owner随后恢复夜间自主推进授权，后继结合既存810相邻证据决定有界学习，不再等待Owner许可；
本条本身不声称810已验收、900后已开训或EMBER已超过MT。

## 212. 810观察确认T近期较好保持，U平台含能力交换；正式后继优先T（2026-09-29）

§14B原件`operator_chain_diagnosis_20260929/adjacent810`。主讨论直接读800新结果行、来源/退出及配对，复算7组统计，
与执行者一致；全新800 NPZ与800 PT继承已审实际消费者，不重复逐tensor或旧原件扫描。
四次外部作业均exit0，成本2.392618928GPUh，新增终值17.798GiB；无失败/恢复、orphan或未计费作业。
T810144/U810132、breadth各6，对同scene MT153仍低；训练来源81846ed3、只读评测d37f647d。
T810→900为+4，R/G/L129/19/15，保留89.6%、Jaccard.791，8-task bootstrap95%[−1.5,4.5]pp。
U为0，108/24/24，保留81.8%、Jaccard.692，区间[−3.25,3]pp；单一训练seed的不确定性未被bootstrap覆盖。
T近期task31+6/16+1，task3−1/6−2；不能再将其近期收益全归于目标碗。U的task31由27→18→14，
900总分132的平台靠其它task补偿，没有恢复450的143。完整各task得失与解释边界见机制§54。

这些结果支持T有限继续学习的优先性，不确定共享A的语义因果，也不宣判U架构无效。
结合§211的匹配功能与尚无验证修正，主讨论按Owner已恢复的自主授权选择§15.4：
T900→1350、U先900→1080停止观察，原架构/监督/优化时钟，完整ECP仍每90步。
U是否追加余270步看完整1080恢复/保持证据，由主讨论明确释放；旧143只作本轮投入参照，非新性能资格。
Sol源码14bac4cd集成b43eef90，主讨论独立14项CPU通过；当前批准执行不等于已启动或取得新性能。
释放后补读100份810 task31谓词：T的9个新成功在810均至少有一个子目标从未达成，U的9个损失在900亦如此；
不支持把近期差额归结为仅末态保持。原270共同成功16例到810为T5/U8、900为T8/U7，保持仍有跨段缺口；见机制§54。

## 213. U1080未出现恢复，本轮停止其追加而不宣判架构无效（2026-09-29）

原件`operator_read_write_learning_20260928/continuation1350/U`。主讨论直接读取新400结果行、旧同臂/MT结果行，
核完整scene/video/RNG前缀配对、每task50不同teacher、实际900父来源/1080 ECP及受控停止记录；
独立复算5组配对与8-task bootstrap，并读三份外部退出/成本。main CPU在同名tmp的
`main_continuation1350_point.py`/`main_U1080.json`。新400 NPZ/PT及8个full RGB、900训练前缀/第三轮teacher/ECP检查，
继承直接审阅过的实际消费者和验收，不重复全扫。源码/训练/评测14bac4cd，原900来源81846ed3；没有重训或科学修复。

U1080为128/400、breadth6；task3/6/11/16/23/26/31/39分别26/1/44/3/0/38/16/0，
suite Spatial/Object/Goal/Long分别27/47/38/16。U900→1080的R/G/L为110/18/22、churn40、
Jaccard.733、保持83.3%，净−4（−1pp），8-task bootstrap95%[−4,1.25]pp。
task3−5、6−1、11−1被16+1、31+2部分抵消，26总数不变但得5失5。
这不是全面能力崩塌；当前新增180步没有给出改善，描述区间也不能证明真实下降或等效。
相对U450为107/21/36、净−15、区间[−10,1.75]pp；相对MT153为95/33/58、净−25、区间[−15.75,3]pp。
U270与1080同为128，但仅保留93/得35/失35，不能由同总分推断学到同一能力。

900→1080训练/物化/official三次均exit0，外部成本2.587438028/.281977217/.953099457，合计3.822514703GPUh。
完整1080 ECP可恢复，但本轮主讨论按§15.4不释放余270步：810/900/1080的132/132/128均未恢复到450的143。
不因剩余预算、对偶形式或“还没完全收敛”自动追加；143只是本轮投入参照，不是新的性能资格线。
这个有限窗口降低原样继续U的优先级，不证明U不可学、S有害或共享A已形成语义度量；
冻结900/450的训练侧视频功能正证据保持，不把闭环阴性改写为接口bug。T沿原已释放范围完成，不受本条扩量。

## 214. T1080与900基本持平，两臂继续学习尚未转为新闭环收益（2026-09-29）

主讨论复用§213 CPU消费者，直接读T1080新400行、同点U及四个旧T/MT原结果，核来源/完成与6组配对；
main原件`main_T1080.json`，新capture继承同一已审实际消费者。T1080=147/400、breadth6，
task3/6/11/16/23/26/31/39为42/5/43/3/0/35/19/0；Spatial/Object/Goal/Long为47/46/35/19。
T900→1080为124/23/24、保持83.8%、Jaccard.725、净−1，8-task bootstrap95%[−1.75,1.5]pp；
相对810为122/25/22、净+3、区间[−1.75,3.5]pp，相对MT为112/35/41、净−6、区间[−6.75,4]pp。
同点T−U为106/41/22、净+19、区间[−1.5,13.25]pp，其中task3贡献+16；不能将其普遍化为共享地址必胜。
900→1080各task净变化仅−1/0/+2/−1/0/+1/−2，近期目标碗/双物能力都没有延续此前净增长；
这180步的总分持平不等于权重或成功集合不变，也不是灾难遗忘证据。

两臂完成的平衡90步在线FM窗，811–900→991–1080为T .095450→.093886、U .094952→.093174；
T1261–1350为.093440。各窗36task各10次，query/flow/权重仍变化，不能称固定query改善、收敛或泛化根因。
实际LR在1200到floor1e−5并延续，900→1350是低LR的有限后段学习，不是重启或足够任意优化的上限测试。
主讨论直接核T900→1350完成记录/2→4拓扑及RNG来源；旧900行前缀和5个ECP消费者验收继承。
450新增更新外部7010.27秒（1.947小时）、7.789184439GPUh；world4各90窗平均宏步14.94–16.35秒，
较旧world2末窗23.85秒实际更快，但视频/节点/学习阶段不同，不称受控线性扩卡实验。T1080 bank/official
外部.295886814/.931973361GPUh；上述训练成本为整个T窗口，两次评测不能重复加账。
T1350完整评测正在原释放内执行，本条不提前选择点或扩展到更长训练。

## 215. T1350单点154略高于匹配MT153，有限后段仍有增益但优势未稳固（2026-09-29）

三组1200新行已全部完成；主讨论在§213–214基础上直接核T1350原400行/7组配对及全部八份外部退出账，
复算一致，main CPU为`main_T1350.json`、`main_continuation1350_cost.json`。新增捕获与训练事件继承已审实际消费者，
没有重复全扫PT/NPZ。最终T1080/U1080/T1350=147/128/154，均breadth6；原件入口为新root的`analysis/report_1350.md`。
T1350逐task为41/6/46/5/0/34/22/0，逐suite Spatial/Object/Goal/Long为47/51/34/22。
相对MT153保留114/得40/失39、churn79、Jaccard.591，净+1（+.25pp），8-task bootstrap95%[−4.5,6.75]pp。
task11比MT多10，但task16少5、31少2、6/26各少1，task3同41；23/39都未覆盖。
因此只能报告这一个匹配面板的微小领先，不能宣布稳定超过MT、整个架构成功或教学视频因果资格。

T1080→1350保留129/得25/失18、保持87.8%、Jaccard.750，净+7，区间[−.25,4]pp；
task6/11/16/31分别+1/+3/+2/+3，3/26各−1。T900→1350保留125/得29/失23、保持84.5%、净+6、
区间[−.75,4.25]pp；相对T450净+32、保持86.9%、区间[.5,16.5]pp，但从T270看仍丢29旧成功。
近期增益不是单独目标碗改善，亦未伴随大范围能力崩塌；仍可能是小幅函数变化引起成功边界交换，不能仅由计数认定学到了新操作语义。
900→1080→1350的148→147→154不是严格单调，更不能按这三个点外推最终分数。

八次GPU均exit0，完整新增14.053326929/24GPUh：训练10.376622467、bank .849007521、official2.827696941。
两次CPU预启动登记修正为0GPUh，原错误/修正留账；无额外episode、孤儿捕获或失败GPU尝试。
新root终值29.5536GiB、strg01 data1终值673571300/1073741824KiB，均为执行者完成时实测，不称连续峰值。
原270批11.158278507、270→900批22.169122590、各诊断批成本分别保留，不因本批结束或低于预算而抹去。

结合§211已有内容相关功能、未验证任何应重训的修正，以及本条新出现的后段闭环增益，主讨论决定只为T登记§16：
原样再观察1350→1800的一个完整50-video轮次，固定1710/1800两点、90步ECP，不自动2250或重启LR。
这是基于新结果的额外有限投入，不是原§15自动晋级，也不以尚未证明不可能作为理由；U停止裁决保持。
若末段仍仅围绕当前能力水平交换，停止原样续训假设的追加，把下一工作转向有实际可失败预测的学习/结构问题。

## 216. T1350低分存在目标物选择错误，不能只用长程保持解释（2026-09-29）

主讨论CPU读取T1350 task3/6/16共150份已完成NPZ及task31/39原谓词，另核两个state0完整RGB与它们真实correct teacher。
task6失败44例中41例目标全程位移<1cm、38例其它物体z升高>5cm（交集37）；task16失败45例中对应35/26（交集26）。
task6升高其它碗29/ramekin9，task16橙汁20/其它6。两例直接视觉核看：演示拿正确碗/黄油，policy分别拿炉子上另一碗/橙汁，
并执行搬向盘子/篮子的行为。几何统计不冒称每例都人工识别了抓取；未读teacher action/state/reward、未新增模型/环境或held梯度。
task31 T1080→1350虽净+3且10个新成功都补上原先未达成的一项，奶酪曾达成后丢失却4→11；保持问题仍存在。
task39在900/1080/1350各50例均两目标从未成立，不能只归为最后关门失败。task23缺开抽屉关节中间证据，不编造阶段定位。

这些事实把部分不足定位为正确教学未可靠转成自身目标对象/关系选择，降低只修末段记忆覆盖的优先级；
不能从行为唯一归因于H表征、M写入、Aq读取或公共β，更不支持即刻换架构/加新loss。
完整样例/数学接口与竞争解释见机制§55；§211的训练侧匹配功能正证据保留，未被held行为反例抹去。
本分析不改变T1800合同、不派新GPU矩阵，只用于之后完整能力和机制判断。

## 217. T1710持平1350，目标选择缺口未消失；原样续训的收益预期降低（2026-09-29）

主讨论独立读取`continuation1800/T/evaluation/1710/correct400`新400行、旧七组T/MT参照、实际bank/ECP/Git及退出账，
复算全部配对与固定8-task bootstrap，结果与Sol一致；同名tmp `main_T1710.json`保留完整统计。
训练/评测fcc23cd15、父1350为14bac4cd，8full+392compact、每task50不同teacher及scene/video/RNG配对一致。
新400 NPZ/PT实际读回和旧1350训练前缀/第四轮事件，继承此次审阅的实际消费者及验收；没有防御性重复全扫。

T1710为154/400、breadth6，task3/6/11/16/23/26/31/39为46/6/43/3/0/35/21/0，suite为52/46/35/21。
T1350→1710保留135/得19/失19、churn38、Jaccard.780、保持87.7%，净0，95%区间[−2.75,3.25]pp。
task3+5、26+1被11−3、16−2、31−1抵消；不是全面崩塌，也没有净收益。
相对MT153为114/40/39、净+1，区间[−5.25,6]pp；相对T900仍净+6、128/26/20，区间[0,3.25]pp。
这些描述区间未覆盖训练seed不确定性；不把接近MT、同总分或跨零分别等同于已超越、同能力或严格等效。

直接读取新150份task3/6/16 NPZ：task6失败44例中目标位移<1cm为40、其它物体升高>5cm为32，交集32；
task16失败47例中对应40/31/31。task3失败4例中对应2/0/0。旧1350的行为缺口未随360步消失，几何量仍不等于逐例语义标注。
task31奶酪曾入篮后丢失11→7，但两目标都曾达成24→22、成功22→21；相对1350得7失8，新增失败7例未曾达成黄油目标。
task39仍50例两目标均未成立。进一步机制边界见§55.4，不从这些行为唯一归因H/M/Aq或公共β。

T1350→1800实际训练已完整exit0，外部6630.88秒/7.367641225GPUh、450更新/50400新query/五个完整ECP；
T1710 bank/official分别.300331052/.985388689GPUh。训练成本属于整个窗口，不能在两点重复计入。
一次CPU事件监听器缺依赖改用已安装host python，0GPUh、未重启实验，原件保留；最终成本仍待1800评测退出。
1710使“本轮还能带来实质净增长”的支持度降低；完成已预注册1800再裁决，不新增旧点搜索/续训或把有效持平当工程故障。

## 218. T1350/1710/1800同为154，结束原样监督追加而不宣判架构上限（2026-09-29）

§16全部800新行完成。主讨论另直接核1800原400行、八组参照配对、bank/ECP/训练与评测fcc23cd15来源，
复算全部得失/8-task bootstrap与五份外部退出账一致；800 NPZ/PT及1350前缀/第四轮/完整ECP读取继承已审实际消费者。
CPU原件为同名tmp `main_T1800.json`、`main_continuation1800_cost.json`；正式入口`continuation1800/analysis/report_1800.md`。
T1800为154/400、breadth6；task3/6/11/16/23/26/31/39=44/7/45/3/0/33/22/0，suite Spatial/Object/Goal/Long=51/48/33/22。
相对1710为135/19/19、churn38、Jaccard.780、保持87.7%，净0、区间[−2,1.75]pp；
相对1350为133/21/21、churn42、Jaccard.760、保持86.4%，净0、区间[−1.75,2]pp。
Sol长消息中该组Jaccard写.7596，实际133/175=.760；统计消费者和本条按原集合值，不改变判断。
相对MT为112/42/41、churn83、Jaccard.574，仍仅+1（+.25pp），区间[−5.5,6.5]pp。
三个154不代表三个相同成功集合，也不是独立seed重复；绝对分数平台与约86–88%的相邻保持同时成立，不能称全面遗忘或稳健超越。

主讨论另读新末点150份task3/6/16行为NPZ与31/39谓词。6的43失败中42例目标位移<1cm、36例同时有其它物体升高；
16的47失败中对应39/33。task31奶酪曾入篮后丢失1350的11→1710的7→1800的4，但黄油末点出现2例丢失，
总成功仍22→21→22。局部保持变好没有转化为整体收益；目标选择/获取缺口依然存在，不能全部解释成末段保持不足。
末点平衡90步在线FM .094229，1350为.093440，五个新增窗口范围.093006–.094310；query/flow/权重变化，
仅说明在线训练读数也未呈持续下降，不能称固定query反事实、数学收敛或能力上限。全过程沿原floor1e−5。

新增五次GPU均exit0，总9.833677124/14GPUh：训练7.367641225、两bank .569593398、两official1.896442501。
最后official于06:53:38 CST退出，实际训练450宏步约1.84小时，bank/official按就绪重叠；一次CPU-only watcher纠正0GPUh保留。
执行者报告新root终值约20GiB、data1终值662.4GiB/1TiB、GPU已释放，不称连续测得峰值；旧批与诊断成本不清零。

按§16事前停止线，本夜原样T监督学习追加结束，U仍在1080；没有自动2250、LR重启/小扫或挑旧点。
此次证据降低“再重复原训练即可获得实质收益”的投入优先级，不证明所有训练方式都无效或当前架构绝不可能超过MT。
没有经过验证的结构/学习修正，不能因平台仓促重训；继续主讨论的特征、任务映射覆盖和共同执行空间分析，
沿最近似历史形成能改变方法判断的最小干预。尚未选定最终checkpoint，也未启动controls/Test/RL。

## 219. 不能用旧rank16或对象未见解释平台；先区分当前M的闭环功能贡献（2026-09-29）

末批结束后，主讨论核当前model/data/spec、coverage协议和完整覆盖审计、旧训练goal元数据，并直接读43/96的实际BDDL。
当前T/U是rank128、O宽256，公共β可表示同类自由MT LoRA；旧rank16不是当前容量事实。
条件差额确受共同A行空间/O列空间约束，但没有实际有益方向落在不可达空间的证据；旧§127的MT低秩近似正例也反对从名义rank直接归因。
butter已在train43入抽屉并关闭、96入托盘，篮子/多物体搬运亦有支持；val16/31留出的是新的对象—受体/组合。
val6有多个碗定位与搬运支持，但cookie-box旁选择留出；39缺的是杯入微波炉的新装置组合，不是从未见杯或关闭微波炉。
这些是规格/训练映射支持，不能说神经特征已学到对应primitive，也不能用更多相同episode替代新迁移关系。
因此本轮不直接扩rank、增加任务池或重开P/I；这些提案仍须改变最近完整负例的预测。

另核强MT300原`run_contract.json`及真实step300 manifest、当前`run.py`/`data.py`：MT每步36task×16query，
300步累计172800；当前T每步4task×28query，1800步累计201600，且每九步无放回覆盖36task一次。
两者的Adam betas、peak/floor及warmup/decay数字相同，但任务出现间隔、每步标签量、实际LR所处阶段与共同β的梯度消费者不同；
不能从总query接近推出优化经历等价，也不能把此差异直接命名为任务冲突或scheduler错误。
相关完整历史§127–133必须一起继承：删其它任务梯度丢掉有益迁移；固定标签4→12条件D54有训练闭环正例及独立面板复核，
随后fresh却未把收益转为held修复（selected142，原C600154/当时MT155），且更多标签B54未单调改善。
这些旧模型仍有主/辅目标及不同参数化，不能当作当前T的大batch直接反证；但足以降低仅凭MT每步覆盖36任务而重开同类改法的优先级。
本次只是读取既有原件/完整论证，未运行新的batch、LR、梯度隔离或lookahead实验。

一个真正未识别、影响完整修正取舍的差别是：已训练T1800的M在held闭环是否帮助公共β。
真实训练优化的是β和M共同组成的策略；β还改变teacher X/H，故完整目标的小梯度不意味着β单独目标也被优化。
900训练面板中完整函数改善、β在3/4 task变差已有实测支持，但不能把它外推到1800的held400。
按active design§17预注册一次仅T1800公共β的冻结400：自身76公共因子，完整M置零，旧full/MT复用，0更新/无Test/无选点。
这区分部署时条件修正的贡献，不识别独立训练效果或视频必要性，不是恢复公共底座课程；结果后仍需具体科学裁决。
新2GPUh/12GiB，预计45–75分钟；唯一Sol实现执行、主讨论并行审阅与原件分析。许可和派发事实以progress为准。

## 220. T1800条件残差有实质闭环贡献，同时保留局部损害；删残差不是性能修复（2026-09-29）

固定§17已完成。主讨论直接读取新400与sealed full/MT原JSON，核每task50视频元数据、完整scene/video/env/policy RNG、
真实1800来源/退出/计费并独立复算；新400 NPZ/PT全量验收继承已审实际消费者，不重复完整扫描。
训练fcc23cd15，真实evaluation-only faac46f8；main9a161650仅补读取时拒绝1710，当前原bank一直为正确1800，未重跑。
一个共享76因子文件，0教学视频值读取；β不是独立训练的语言baseline。

β100/400、full154、强MT153；breadth5/6/6。按task3/6/11/16/23/26/31/39，β为6/3/29/0/0/41/21/0，
full为44/7/45/3/0/33/22/0。full−β保留75/得79/失25、churn104、Jaccard.418994，净+54；
固定8-task bootstrap95%为[−2.25,33.5]pp，不能把单种训练起点/有限任务的净收益外推为普遍保证。
β−MT为75/25/78、净−53、区间[−30.75,0]pp。full对MT仍只+1，不把较弱删减参照替代强MT。

主讨论另有目的地读取200份β行为NPZ和50份full task26 NPZ；full3/6/16沿既有实际读取结果复用。
β的task3全部44次失败中目标位移<1cm，full从中恢复38次成功且保留原6次；不是只改参数/动作而无控制作用。
task6/16的β失败中目标几乎不动为46/47、50/50，目标不动而其它物体升高>5cm为39/47、45/50；
full对应42/43、39/47及36/43、33/47。删除M没有解除既有目标选择缺口，full还部分缓解了它。
task26则full相对β得3失11；β9次失败中8次目标升高>5cm，full17次失败全部如此，不能将这一局部损害也统称为未动目标。
task31得12失11、共同成功仅10，总分21→22掩盖明显交换；39在三模型均无任一目标曾成立。
几何采用真实body_xpos，只描述行为，不声称所有失败的视觉语义或具体神经层因果已定位。

判断：保留有用的条件通路，降低“去掉/普遍压小M就会整体更好”的优先级；不把β弱自动解释为公共参数训练失败。
完整FM只监督共同策略，β还形成teacher H/X，条件残差可承担通用修正；冻结删除只识别当前部署贡献，不识别独立训练责任。
900训练面板的匹配/近邻正例仍成立，但本次不证明held视频匹配或时间顺序必要性，也不授权task门控/挑视频/按task拼接β和full。
机制与下一学习假设见机制§56；任何修正须由完整策略的实际收益裁决，不能以β分数或内部指标改善代替。

一次4卡外部894.1536秒、0.993504002/2GPUh，exit0，07:39:19 CST结束；12workers全0，0失败/孤儿，8full+392compact。
Sol报告新产物约818MiB、GPU已释放，未量连续存储峰值。原件`operator_public_beta_diagnosis_20260929`，
主讨论CPU分析`main_public_beta_analysis.py/json`在既有同名tmp；无新增训练/Test/RL或额外GPU诊断。

## 221. 匹配公共风险短窗已产生功能变化，正式闭环仍待整组验收（2026-09-29）

§18两臂各90新更新已exit0：control world2、public_aux world4，真实1800父状态/原Adam/绝对LR保持，
360个新task-teacher-query-flow事件跨臂逐项相同。主讨论直接核新90行、合同/退出/1890训练状态与首末Adam step；旧1800前缀继承已审消费者。
所有更新grad finite、0次clip，公共与全部Writer分组均有信用；不能把本次作用归于全局clip更强或公共项未接通。
主讨论实际读取36份PT，固定4train×2teacher/28query/flow的公共FM差aux−control平均−.00069308，完整FM−.00010567，
分别4/4task、8/8条件更低。原件FM与行记录最大误差8.26e−9；小差额不来自该聚合读回容差。
但task20两teacher前5步FM均比control高，平均+.00047293；公共改善伴随条件函数共同调整，不能据代理值宣布修正成功。
精确预测分解与反例见机制§56.4；尚无两组新official完整分数/资格、held视频必要性或独立语言baseline结论。
已结束训练成本1.149129799+1.871410659GPUh，36功能前向.041878113GPUh；这是已完成部分，物化/official与最终总账另待完整回报。

## 222. 辅助臂1890单节点略低于父状态，匹配参照尚待回报（2026-09-29）

主讨论直接读public_aux新400及sealed父1800/MT JSON，核全部scene_reference、语言、teacher/ordinal、env/policy RNG，
逐task50视频、8full+392compact、实际bank到自身public_aux/loss/1890训练来源及外部完成账。训练/评测均9801641d，
父fcc23cd15；实际freeze不包含后加的main臂身份窄修，但本bank/train身份直接核为正确，无需重算。
全量新NPZ/PT验收继承已审实际消费者，未逐tensor重扫或读取未完成control；CPU计算在既有tmp的`main_public_function_official.py/json`。

aux148/400、breadth6；task3/6/11/16/23/26/31/39为43/5/43/3/0/36/18/0。
相对父154为136/12/18，churn30、Jaccard.819277、保持88.3%，净−6，8-task bootstrap95%[−4,1.25]pp；
相对强MT153为111/37/42、净−5、区间[−7,4.5]pp。小幅净退步不等于全面遗忘，也未改善绝对能力或覆盖。
task26得6失3，但task31得4失8；31两谓词各自曾成立的episode由父24降19，最终同时成立22降18。
task39有1例仅关上微波炉、杯入从未成立，完整成功仍0；不能隐去这个局部变化，也不能称解决了组合任务。

固定四train任务的β/full FM改善尚未转为本节点的净closed-loop收益，主要aux−control比较仍未知。
另保留不利范围：90步相同事件的在线完整FM均值control .09276470、aux .09277431，并非全训练流均改善；
这跨越模型更新，不能替代固定终点面板或反向声称终点训练总体变差。四task正例不推广成全36task结论。
只有完整control回报后才裁决额外公共风险的作用，不用尚未完成的分数、不按这次负差额改目标或加预算。

aux bank exit0 .268650979GPUh；首次official 4workers/card启动OOM及一项framebuffer错，exit1、0有效行/0NPZ/PT，
失败原件保留于`public_aux/evaluation/1890/failed_attempt0_4workers_gpu01`，搬移记录与.071013475GPUh已直接核。
同固定400条件恢复3workers/card，9workers全0，09:26:34 CST退出、.941802990GPUh。不是按结果选重跑，成本不清零。
训练/公共面板沿§221；control及整批总账尚待回报，未把成功canonical树的0失败写成整个批次无失败。

## 223. 匹配参照158、公共辅助148：结束这项修正，不把小幅新高当续训保证（2026-09-29）

§18整批已完成，覆盖上节的等待状态。主讨论独立读两组新800 JSON、sealed父/MT、两bank与训练臂/损失身份、
完整scene/video/env/policy RNG及九份外部退出账；实际NPZ/PT内容验收继承已审消费者，36固定训练预测PT与学习事件沿§221。
control158/400、aux148，父154、强MT153，breadth均6；control按3/6/11/16/23/26/31/39为42/4/45/5/0/36/26/0，
aux为43/5/43/3/0/36/18/0。所有正式行每task50合法不同视频，scene/视频映射未改。

| 候选−参照 | 净成功数 | 保留/得到/失去 | churn | Jaccard | 固定任务簇95%区间(pp) |
|---|---:|---|---:|---:|---|
| control−父1800 | +4 | 141/17/13 | 30 | .824561 | [−2,4] |
| control−MT | +5 | 117/41/36 | 77 | .603093 | [−3.75,7] |
| aux−control | −10 | 136/12/22 | 34 | .800000 | [−6.75,.5] |

control保留父成功91.6%，原目标后90步有有限净收益，不能抹去158；但它没有新覆盖，6由7降4，23/39仍0。
aux相对control的净差−10中，task31贡献−8，task11/16各−2，3/6各+1，26相同但仍有得2失2。
31的control26次成功里，aux保留14、丢12，另得4；12个丢失中10个只曾奶酪入篮、1个只曾黄油、1个均未，
其中4个奶酪曾入后又丢失。它同时有第二对象获取与局部保持缺口，不是统一的末段遗忘。
两谓词各自曾成立由control26到aux19，最终同时成立26到18；不把这项行为差额直接命名为教学M的时间覆盖。

公共风险已接通，公共训练函数改善、所有更新未clip，仍未在匹配闭环得到收益。因此按事前§18.3关闭该固定系数/父状态/
低LR短窗修正，不再延长、扫lambda/重启LR/冻结β或直接fresh；不以区间跨零宣称两者等效，也不扩大成所有公共辅助目标必然有害。
158是本次原目标参照的真实单点，尚非稳健超过MT/相邻资格或正式selected；不足以推翻已测多窗口平台而自动恢复长续训。
视频通路的900匹配功能和1800闭环净贡献仍有效，本次不是“视频整体无用”或唯一内部根因的证明。

control首次official在300完成行后RGB scene一致性断言exit1；主讨论读失败/两次invocation/只读queue及28份原完成shard，
直接核这300行与最终400逐行相同，恢复只补100行；41份未引用NPZ与正式行路径不相交、0孤儿PT，未将孤儿结果择优合入。
原失败、.839743196GPUh和恢复.180654835GPUh均保留；aux失败沿§222。完整9尝试含2次失败，总5.630391058/8GPUh，
最后official于09:40:35 CST exit0。Sol报告root终值19G、data1 quota682.4G/1T、GPU释放；未测连续高水位。
原件`operator_public_function_pilot_20260929/analysis/final_report.md`，主讨论CPU分析沿同名tmp。没有额外1890后训练、Test、RL或视频controls。

当前仍缺本架构对已见任务的完整闭环证据：四task固定FM正例不能代表36task，也不能与8held低分自动拼成“只差迁移”。
历史多版train96已有训练获取高而validation低、或两者都低的不同实例（§28–34/53–56/100–102），不能代替当前读写图。
下一项仅冻结原预登记T1800与强MT300，按新§19读出实际36训练任务的配对闭环；不选择新checkpoint或增加训练。
它将限定已见任务获取与新关系迁移各有多大缺口；不会单凭一个均值确诊数据、编码器或Compiler。

## 224. 1890后首个90步节点为156，保留后段观察而不提前判定最高点（2026-09-29）

Owner午前明确恢复§20有界续训。主讨论独立读T1980完整400、sealed control1890/MT及来源/完成原件，
逐行完整scene_reference、exact语言、teacher/ordinal、env/policy RNG配对成立，50不同视频/task。
训练来源e2afbfd7，物化3ba62976，official cb535c1e；全NPZ/PT沿已读实际verify_official_raw_group.py与exit0日志继承，未重扫。

T1980=156/400、breadth6；按3/6/11/16/23/26/31/39为46/5/40/4/0/38/23/0。
相对1890=158：R135/G21/L23、净−2、churn44、Jaccard .754190，保持85.4%，8-task bootstrap95%[−4.25,3.00]pp。
相对强MT153：R117/G39/L36、净+3、churn75、Jaccard .609375，区间[−4,5.25]pp。
1890→1980的task3/6/26净+4/+1/+2，11/16/31净−5/−1/−3；23/39仍0。不是全任务同步恶化，
也不是全面遗忘；只有一个小幅回落点，不能宣布持续下降、过拟合或158是架构最高值。现有差额不触发Test。

训练已从唯一control1890父状态续至2340，外部12:56:07 CST exit0，7137.068秒×4=7.930075568GPUh。
main核原1890 metrics字节前缀、连续2340行、新450行各112query/full/floor1e−5及五份90步ECP manifest/大小；
完整optimizer数值不重复全扫。其余四个预登记official节点仍待完整结果；已生成/启动节点全部计入，不因首点回落丢弃不利观察。
初始1980 bank timeout保留379条件后原freeze续行、总.366220899GPUh，official1.213517775GPUh；失败不清零。
main复算在既有tmp `main_1980_seen_acceptance.py/json`。有限多点最高值须保留选择乐观性，任务簇区间不覆盖训练seed不确定性。

## 225. 已见任务闭环T105/144对MT93/144，链路已获得任务内能力但不等于迁移充分（2026-09-29）

§19固定原T1800/fcc23cd15与原MT300/3ebb979b，36训练任务×4初态，两个冻结模型共288完整行。
主讨论直接读两臂results/contract/completion、逐行scene/video/env/policy RNG并独立复算全量与子集；
canonical场景cb535c1e统一重生后严格恢复，保留原物理状态/teacher映射，旧权重由显式lineage引用，未重新训练或物化。
全288 NPZ/PT及72 full RGB继承已完整阅读的verify_canonical_288.py实际消费者及输出，0孤儿，不称全tensor或跨硬件bitwise验证。

| 范围 | T1800 | MT300 | T−MT保留/得到/失去 | 净差 |
|---|---:|---:|---|---:|
| 全36task | 105/144 | 93/144 | 83/22/10 | +12 |
| 原24训练task | 61/96 | 46/96 | 39/22/7 | +15 |
| 额外12支持task | 44/48 | 47/48 | 44/0/3 | −3 |

全36task breadth为33/30，churn32、Jaccard .721739，36-task簇bootstrap95%[0,17.36]pp。
原24中11task净提升、4task净下降、9task同分；支持12中3task各失1，其余同分。四suite T/MT为18/13、17/11、16/14、10/8，
额外libero90为44/47。原件全per-task与分项配对保留，不能只报总体正差掩盖支持集的负差。

相较只在专家query上降低FM，现在确认固定生成LoRA在不少已见任务的自身闭环状态也能形成实际成功，
削弱“整个条件链路没有学会任务内功能”的全局解释。原24仍有35/96失败，获取/执行缺口并未消失；
与held400属于不同任务和难度面板，不能相减估算唯一的泛化损失或认定所有不足都来自新组合。
这提高研究条件函数跨任务复用的优先级，不直接定位H、K、M或Ah，不证明增加任务必然有效，
也不证明视频时间过程必要：本面板没有同模型视频干预，teacher池已用于训练，四初态一次seed不等于新视频/完整初始化泛化。
机制§57.1记录两条轴的更新；不因这项正例重开公共风险修正、更多面板或新架构。

14份外部退出回执（含CPU注册与所有失败/恢复）由main重新累计1.070418068GPUh，低于3GPUh；
旧scene、失败队列/两次环境复现和canonical首试EGL映射错误全部保留。原件入口
`/data1/user/ymdai/ember_runs/operator_seen_task_diagnosis_20260929/analysis/canonical_288_validation.json`及同目录pair_breakdown/验证脚本。
§19已收束，不新增计算，其临时第7卡许可随完成结束；独立§20继续原已授权节点，不由本项直接选点或触发Test。

## 226. T2070降到152：连续两次小幅回落集中于task31，相邻保持仍为90.4%（2026-09-29）

主讨论独立读取2070完整400与sealed1980/1890/MT，复算所有scene_reference/语言/video/env/policy RNG配对、50视频/task及来源。
训练e2afbfd7、bank/official cb535c1e，T2070=152/400、breadth6；task3/6/11/16/23/26/31/39为45/5/42/5/0/37/18/0。
相对1980：R141/G11/L15、净−4、churn26、Jaccard .844311、保持90.4%，8-task簇95%[−4,1.25]pp；
相对1890：137/15/21、净−6、Jaccard .791908、区间[−6.5,2.25]pp；相对MT：114/38/39、净−1、区间[−5.75,5]pp。

1890→1980→2070为158→156→152，确实已观察到两个相邻小幅回落，削弱本窗口持续净增益预期，
但不证明过拟合、1890是架构最高点或未来不可回升。训练早已完整到2340，剩余三个预登记评测全部计入，
不因回落截去后段或提前选择1890，不自动追加2340后训练；现有结果不触发Test。

这次1980→2070净−4中，task31贡献−5，其他任务合计+1。task31由26→23→18，2070相对1980得3失8；
失去的8次中7次奶酪曾入篮、黄油从未入，1次相反；前7次中3次奶酪最终也丢失。
两目标各自曾完成的行数同样为26→23→18。最早可见不足包括一个目标未完成，并伴随部分局部保持失败；
这些谓词不能区分目标选择、接近、抓取或放置，更不能直接命名为M覆盖、特征遗忘或梯度冲突。

2070首轮116行后严格RGB断言exit1，恢复13:34:15 CST exit0。main只读queue并核故障前13份完成shard，
116行与最终400逐行相同；37份未引用NPZ排除在正式路径集合外、0孤儿PT。全400新NPZ/PT内容继承已审实际消费者，未重扫。
bank .302521797、首official .506051816、恢复 .559228714，共1.367802328GPUh，失败不清零。
原件`continuation2340/analysis/raw_integrity_T2070.json`及正式results/queue/退出账；独立计算保存在既有tmp `main_T2070_acceptance.json`。

## 227. T2160/2250回升到155/156，近期为成功集合交换中的平台波动而非持续单调下降（2026-09-29）

主讨论独立读完整400及2070/1890/MT参照，全部scene_reference/语言/teacher/ordinal/env/policy RNG配对成立；
各task50视频，训练来源e2afbfd7、物化/official cb535c1e。新400 NPZ/PT内容沿已审实际消费者继承，无新扫描。
2160为155/400、breadth6；task3/6/11/16/23/26/31/39为44/8/43/3/0/36/21/0。
相对2070：R137/G18/L15、净+3、churn33、Jaccard .805882、保持90.1%，8-task簇95%[−1.5,3.25]pp；
相对1890：138/17/20、净−3、区间[−4.25,3]pp；相对MT：113/42/40、净+2、区间[−5,5.75]pp。

158→156→152→155中，前两次回落没有继续单调延伸，但本点仍未形成新高或明显超过MT。
相对2070新增18次中14次曾在1890或1980成功，只有4次在这三个早前观察点均未成功；这是案例复现统计，
不构造checkpoint union，不把14次恢复或4次首次观测自动解释成永久恢复/学会了新语义。
本点task6/31各净+3、11净+1，3/16/26净−1/−2/−1；task31得7失4，成功18→21但仍低于1890的26。
31两目标各自曾成立22次、最终同时成立21次，保留局部保持缺口；23/39仍0，没有新增任务覆盖。
现有证据较支持平台附近的非单调交换，不能称已证明收敛或全局最高；剩余2250/2340完整计入后再按原规则选点，不由回升自动续训/Test。

本节点13:53:45 CST一次exit0，3326.870秒×1=.924130466GPUh；0失败、0未引用NPZ/PT。
原件`continuation2340/analysis/raw_integrity_T2160.json`、official results/contract/completion及外部退出；
主讨论独立CPU复算在既有tmp `main_T2160_acceptance.json`。

后续T2250完整400已由主讨论独立读行/配对/来源/退出并复算：156、breadth6，task3/6/11/16/23/26/31/39为42/7/44/3/0/35/25/0。
相对2160：R139/G17/L16、净+1、churn33、Jaccard .808140、保持89.7%，8-task簇95%[−1.75,2.75]pp；
相对1890：140/16/18、净−2、区间[−2.25,1.75]pp；相对MT：116/40/37、净+3、区间[−4.75,6.25]pp。
task31得9失5、净+4至25，两个目标各自曾完成25次、同时完成25次；其他任务合计−3，因此局部恢复未形成总体新高。
完整观察序列158/156/152/155/156仍在窄范围波动，四个新增点均低于原158；不把连续两次回升转述为已证明长期增长。
仅2340还未完成，其完整结果仍纳入事前选点，不提前选择或释放候选controls/Test，不扩训练窗口。
本节点13:56:30 CST单次exit0，3333.369秒×1=.925935902GPUh，0失败/孤儿；训练e2afbfd7、eval cb535c1e保持。
全400 NPZ/PT内容继承已审消费者；原件`raw_integrity_T2250.json`和对应official目录，独立复算在既有tmp `main_T2250_acceptance.json`。

## 228. T2340取得161有限新高，Owner明确要求四卡继续学习而暂缓其它后段（2026-09-29）

§20最终五个新节点全部完整，原1890与1980/2070/2160/2250/2340为158/156/152/155/156/161，breadth均6。
主讨论直接复算2340全400、全部旧观察点/MT的scene_reference、语言、teacher/ordinal、env/policy RNG，50视频/task；
确认T/full、真实e2afbfd7训练来源及cb535c1e读取身份。全新NPZ/PT验收沿已审实际消费者继承，不重扫旧捕获。
2340按task3/6/11/16/23/26/31/39为45/6/45/5/0/36/24/0，suite为51/50/36/24。

| 对照 | 净成功数 | 保留/得/失 | churn | Jaccard | 8-task簇95%区间(pp) |
|---|---:|---|---:|---:|---|
| 2340−2250 | +5 | 141/20/15 | 35 | .801136 | [−.5,3] |
| 2340−1890 | +3 | 139/22/19 | 41 | .772222 | [−1,2.75] |
| 2340−MT153 | +8 | 117/44/36 | 80 | .593909 | [−3,7.5] |

2340为预登记规则下本窗口最高候选，但未发布selection文件。相对1890净+3来自task3+3、6+2、31−2，其他同分；
相对MT主要为11+9、3+4，被16−5、6−1部分抵消，26+1、31同分。末段确有新高，不能抹去；
450新增更新后的净收益仍有限，23/39零覆盖不变，多次观察后的最高值亦有选择乐观性，不能直接称稳健超越或全局最高。

主讨论直接重算整批外部回执：续训2340为14.551285328GPUh，含全部失败；§19为1.070418068，合计15.621703396，
无已启动未结job。此前1980物化timeout/元数据错记、2070恢复及37未引用NPZ全部保留，不重新生成已验有效结果。
独立汇总在既有tmp `main_final2340_analysis.json`，原件`continuation2340/analysis/external_cost_ledger.json`及各node正式目录。

Owner先要求分析讨论，随后明确优先让Sol四卡续训；后者仅恢复继续学习及原间隔correct观察，不释放视频对照/Test。
按新§21从唯一完整T2340接到2790、每90步完整ECP/paired400；这是Owner明确选择继续观察，不伪装成已经证明续训必然有效。
未发送旧拟议的selected视频任务、未发布selection，无新增Test/controls结果。后段其它选择留待讨论，主讨论先完成该明确执行授权。

## 229. 续训期间的实际学习/行为分析：U拟合更好仍未转成held优势，近期增益包含旧成功恢复（2026-09-29）

主讨论只读已完成T2340、U1080和旧V/L450 metrics，以及sealed1890/1980/2070/2160/2250/2340完整结果；
没有读取正在运行2790的metrics/半批分数。CPU分析位于既有tmp `operator_chain_diagnosis_20260929/`的
`main_learning_stability_2340.py/json`；模型/环境/GPU新计算为0。

实际T/U前1080步共4320个条件的task、teacher、28 query episode/frame、flow seed和应用LR逐项匹配。
451–900的1800对中U−T平均在线FM为−.0004883269，60.5%条件更低；901–1080的720对为−.0006587467，67.22%更低；
两窗均33/36task的均值更低。配合既有固定终点面板，降低“U仅因训练拟合不足而较差”的解释，
但这些是持续更新模型的在线观测，不能替代冻结风险，也未证明T的共享约束就是唯一迁移根因。

T/U271–450实际LR从.000290754降到.000245745；旧V/L记录的是after-step LR，约.000290601→.000245406，
字段时点不同，不报作恢复时钟错误。当前T1351–2340实际LR恒1e−5，明显低于旧V退化时段。
T2340/U1080/旧V与L450全部已完成日志均无norm>1的clip触发；低LR与训练阶段是稳定性比较的竞争解释，
但旧V270→450的退化与现T/U同期改善说明不能将全部差别都归给后期LR。旧场景不与现400假称严格配对；
旧L450的24行初位勘误继续见§186，不据当前比较重新包装为全400配对。

六个90步观察点158/156/152/155/156/161中，112个case始终成功、210个始终失败，其余78个有变化。
本序列曾成功190个只是案例分析集合，不能当作可部署policy/选点或能力分数。
1980→2070新得11例中9例此前成功过；2070→2160的18例中14例、2160→2250的17例中15例、
2250→2340的20例中17例也曾在本六点序列的更早节点成功。最后3个首次成功case为(task6,state1)、(16,16)、(26,16)，
“首次”仅限1890–2340六点，不称全历史首次/学会了新任务。近期增长含能力恢复与首次观测，不全是新覆盖或持续衰退。

进一步针对低分task6/16读取T1890与2340的200份既有NPZ位置字段，另读同场景MT的100份参照；
这是行为解释而非完整性复扫，未重跑评测。`main_recent_behavior_2340.py/json`与`main_mt_behavior_reference.py/json`保留逐例数据。
下表后三列统计失败样本；“不动”为目标最大位移<1cm，“抬起其它”为非目标/目的物体z增量>5cm，不推断意图或已夹持。

| task / 模型 | 成功 | 失败 | 目标不动 | 目标不动且抬起其它 |
|---|---:|---:|---:|---:|
| 6 / T1890 | 4 | 46 | 42 | 36 |
| 6 / T2340 | 6 | 44 | 42 | 33 |
| 6 / MT | 7 | 43 | 38 | 34 |
| 16 / T1890 | 5 | 45 | 39 | 29 |
| 16 / T2340 | 5 | 45 | 37 | 26 |
| 16 / MT | 10 | 40 | 35 | 21 |

task6的T2340失败中25次另一黑碗抬起、8次瓷碗抬起；MT相应2/32次。共同目标不动的33例中，13例为T动另一黑碗而MT动瓷碗。
这说明目标实例获取与分支语义调用值得解释，而非只有动作末端精度；同时两者共享大量困难，不能将其独占归罪于视频Writer。
两点task31成功26→24，奶酪曾达成后丢失2→4；task39的50例在两点均未触发杯入微波炉/关门任一终点谓词。
轨迹和阈值不能定位教学编码、M写入、自身地址或低层接触的唯一根因。机制解释及旧源码实质比较见机制§58。

Owner已授权的唯一新增冻结诊断为active design§22：教学native中的公共β替换source identity，保留最终执行β/Writer，
T/U900与T2340固定4train×2teacher，36新FM前向，0更新/环境/held标签，1GPUh/4GiB。
合同已push并以Steer送Sol当前轮，接口接受不等于已执行；此处无该项结果或修正成功声称。

## 230. 冻结教学读取诊断完成：有益适配较小，source-native仍保留多数条件功能（2026-09-29）

§22 root=`/data1/user/ymdai/ember_runs/operator_chain_diagnosis_20260929/native_reader_transport`。
唯一attempt exit0、197.173491秒/.054770414GPUh，36新FM、8共享source-native特征，约2.1GiB，0更新/环境/held标签。
实际数值源e2afbfd7，T/U900来源81846ed3、T2340来源e2afbfd7；只更换教学read_native_video的公共参数为source identity，
Writer公共A/B0、U的S、Value、最终执行β保持。公共源读取的8视频响应跨三模型复用；900正常full16及β8沿§14A引用。
GPU02六卡正式任务占满，Sol按短分析例外在GPU01:0使用临时第七卡后退出释放；原续训/official没有中断。

主讨论直接读实际脚本/合同/退出，以及新36与所需旧28份prediction/target/time切片，确认该科学干预，
独立复算FM/前5/后45及误差分解，scalar最大差1.15e−8。其余M/全部特征的形状与有限值验收继承Sol实际消费者，不逐tensor复扫。
CPU过程在现有tmp的`main_native_reader_analysis.py/json`，另8对H的实际Value输入分析为`main_native_feature_transport.py/json`。

| 冻结模型 | 正常full FM | source-native full FM | 自身公共β FM | 干预−正常 | 变差/改善条件 |
|---|---:|---:|---:|---:|---:|
| T900 | .11122231 | .11219630 | .13382817 | +.00097399 | 6/2 |
| U900 | .11016959 | .11124680 | .13367383 | +.00107720 | 7/1 |
| T2340 | .11080591 | .11195396 | .13386420 | +.00114806 | 8/0 |

全部24个干预后条件仍优于自身公共β，其中21/24优于强MT；强MT同面板平均.11741647。不能称独立语言baseline或给知识作比例归因。
正常−公共功能量级约−.023，教学读取干预损伤约+.001，支持当前适配有用但未支持它单独解释条件分支大部分功能。
T/U900均值损伤相近且U仍有更低train FM，不解释T的held优势，也不证明从头冻结reader会保持当前能力。
反向条件保留：T900 task0/40为−.00007538、32/43为−.00003805，U900 task32/43为−.00087926；微小差额不夸大。

T2340前5平均干预损伤+.00168661、后45+.00108822，但3/8前5改善；T/U900也各3/8前5改善。
精确输出误差分解最大误差3.28e−17；T2340均值线性项−.00048886、二次项+.00163692，5/8线性项有利而有限差额有害。
这不是参数一阶近似或梯度冲突，也不能从全50步FM直接推十步采样后前5实际动作的闭环排序。

8对已存H按实际RMS单位化后，差RMS为.0848–.1216；时间差分的改变量/正常时间差分RMS为.425–.830，
不是“全部适配被差分抵消”。但未将变化的轴识别为目标语义，P/C/D/O、递归保持与自身A h的具体分工仍未隔离。
同一固定面板T900→T2340平均full仅改善.00041640；task12全50变化接近零，task32全50改善但前5变差.00100569。
这张4train面板不能替代整个学习曲线或解释held148→161。完整判断见机制§59；本诊断收束，不据此改架构/损失或追加局部矩阵。

## 231. 单项收束不等于研究只能等待：训练功能排序随flow时刻改变，进一步检验条件作用的实际生成传递（2026-09-29）

Owner指出已有解释仍有重要未知，主讨论此前结束独立研究过早；继续此前并行分析授权，§22本身不重开。
新增CPU分析只读§14A/22已存prediction/target/time，共40个唯一预测原件，0模型/环境/GPU。
实际脚本及逐query记录在既有tmp `operator_chain_diagnosis_20260929/main_fm_time_prefix.py/json`。
固定四train任务各28query，先对同query两teacher的损失取均值，不平均模型或prediction；三模型与MT/β配对身份沿旧原件。
tau分组事先固定[0,1/3)、[1/3,2/3)、[2/3,1]，各组24/40/48个query；各task的组内样本数不同。

| U900减T900的FM | 低tau | 中tau | 高tau | 全部112query |
|---|---:|---:|---:|---:|
| 完整50 | +.00027267 | +.00047417 | −.00298781 | −.00105271 |
| 前5 | +.00294264 | +.00087944 | −.00422081 | −.00086427 |

高tau前5的U优势在4/4任务组内成立；低tau前5的U劣势在3/4成立，中tau前5的净劣势主要由task32贡献，
不能把三组均值讲成所有任务同向。各组先task等权再平均仍保留表中符号，排除仅由组内任务权重造成总体反号的简单解释；
但各tau对应不同query/阶段/初态，未进行同query的time干预，也未建立噪声时刻引发闭环排名的因果结论。

T2340完整相对自身β的前5 FM在三组均改善（−.01919/−.03554/−.02813），不支持“条件作用只有低噪声标签提示下才有用”。
相对MT的完整50三组均改善，但前5为+.002436/−.009362/+.000942，显示此前全50改善不能无条件外推到所有前缀/时刻。
这是同一训练小面板的条件描述，不把专家动作MSE当多模态动作正确性或部署性能。

已有“U平均训练拟合更好却held更差”保留，但其解释必须细化：所学函数的优势可能随suffix状态/time改变，
不能只在“训练拟合vs任务泛化”二分法中停留。当前尚未知道这些差异会在真实10步自身生成中放大、抵消或消失。
因此新§23固定原T/U900与T2340/各自β/MT、同真实观测与noise，读取实际10步生成，并在β路径上额外读full速度；
以有限精确分解区分条件直接作用与自身suffix反馈，不称环境反馈，不新增held标签或正式controls。
独立1GPUh/4GiB合同已push并Steer接口接受；尚无该GPU结果。具体方法取舍仍依该相对证据与历史辅助阴性，
不自动改FM权重、sampler或恢复旧endpoint路线。

## 232. T2430降至147：回落集中三task并涉及既有稳定成功，先解释具体失效而不从单点判收敛（2026-09-29）

Sol交付完整2430后，主讨论直接读取本组400原行、sealed2340/MT及1890–2250完整成功集合，
逐行核scene_reference、teacher/condition/video ordinal、exact语言、env/policy seed和共同长度的policy noise序列。
直接核run_contract/完成记录/外部退出与训练源7a0fdcba、evaluation-only源3cb113c6；新capture内容检查沿Sol实际消费者继承，
未再加载400 NPZ/PT、未读后续半批。独立过程在既有tmp `main_2430_analysis.py/json`，无模型/环境/GPU。

2430=147/400、breadth6，per-task3/6/11/16/23/26/31/39为45/6/43/5/0/31/17/0。
相对2340=161：R135/G12/L26、净−14、churn38、Jaccard.78035，保持83.85%；
相对MT153：R108/G39/L45、净−6、churn84、Jaccard.5625。执行者固定8task bootstrap区间分别
[−7.25,−.4875]pp与[−7.25,+4.75]pp，继承其已验方法，本次独立复算行级配对/得失而未重抽样。

净下降14全部来自task31−7、26−5、11−2；其它task净0但仍有交换。task31得5失12、task26得2失7、task11得0失2。
26个丢失中8个在1890–2340六点全部成功，分别为(11,14)、(16,8)、(26,13/17/33)、(31,11/39/49)；
因此不能把它全部解释为2340偶然取得的边缘高点。12个新得中仅(31,12)此前六点未成功，其余为旧成功恢复。

直接读阶段谓词：task31丢12例中10例本次黄油从未入篮，1例两个谓词分别曾成立、1例只有黄油曾成立；
终态6例仅奶酪、4例两物均未达成、2例仅黄油。全50例奶酪曾达成后终态丢失由4→7，
两物各曾达成的例数24→19、最终同时达成24→17；“各曾达成”不等于曾同时成功。
task26丢7例与task11丢2例，本次均从未达到各自目标谓词。这使本轮变化同时包含目标获取不足和局部保持，
不能统一叫末段遗忘、视频递归覆盖或动作精度；未由本结果指定某层/某项loss为根因。

新的90步跌幅超过此前1890–2340的局部波动，需修订稳定性表述；churn38没有同倍激增，净降来自损失与收益更不平衡。
单点尚不证明持续下降或架构最高点，按Owner已明确§21继续原范围，独立§23也不等后续节点。
bank0接口读取失败exit1/.003726749GPUh原件保留；bank1窄修后exit0/.279796743，official一次exit0/.922386340。
三项合计1.205909832GPUh；当前整批未结束，不把已退出作业账或本节点账称最终账，不重测MT。

## 233. 语言确实进入动态写入及实际读取算子，不能把未证语义简化成语言被忽略（2026-09-29）

Owner要求离开用餐期间自主深入架构。主讨论只消费§14A已完成T/U900的各4种target语言×8条固定视频，
读取64份PT中的实际H、Q8/V8/out的M及正确条件query输入，并从各自ECP读这3处A。
实际脚本/数值为既有tmp `main_language_video_operator.py/json`；0新模型/环境/GPU，无held标签或选点。
这不是重验旧原件完整性，而是利用原完整交叉设计回答此前没有量化的语言作用问题。

对同一video的4种语言，原始RGB/帧索引保持不变。实际RMS归一化后的跨帧差分ΔH中，
随语言变化的中心化分量能量占四语言总ΔH能量，T的8视频为42.82%–61.55%，U为43.08%–61.74%。
这是有限反事实输入组的平方范数比例，不是语义信息比例、训练增益、因果中介份额或语言“解释了多少成功”。
它排除了在这组实际特征上“语言仅添加一个被时间差分完全消除的常量”的说法；不证明正确对象关系被编码。

进一步不直接比较有坐标歧义的M范数，而在固定自身query地址上度量其线性作用。各site用该臂8个正确条件
保存的完整28×50 query等权构造C=E[(Aq)(Aq)^T]，固定C后，||M||_C²=tr(MCM^T)就是这些query上
条件预激活的平方和均值。平衡4×8表精确分成总体均值、语言主项、视频主项和交互；下表分母为去总体均值后的能量：

| 臂/实际site | 语言主项 | 视频主项 | 交互 |
|---|---:|---:|---:|
| T/Q8 | 13.53% | 45.50% | 40.98% |
| T/V8 | 20.60% | 38.50% | 40.90% |
| T/out | 21.61% | 33.96% | 44.43% |
| U/Q8 | 20.56% | 35.12% | 44.32% |
| U/V8 | 14.69% | 53.77% | 31.55% |
| U/out | 15.49% | 39.05% | 45.46% |

均使用对应模型自己的真实query分布，不能把T/U不同C的绝对数当公平性能比较；也不把3site外推为38层因果归因。
该分解的数值恒等式相对误差最大4.54e−15。由于真实query固定，交互不能由换语言时query也被换掉解释；
它表明视频作用会随编译目标语言而改变，非“一个视频项加一个与视频无关语言偏置”的简单结构。
但预训练原生非线性及共同适配也能产生交互；交互存在不证明组合泛化或有序操作知识。

在实际正确query hidden完全固定时，只将out的M换成同一RGB、另一编译语言的已存M，
实数算术中的有限预测差为(M_otherL−M_correctL)Aq。T的24个替换全部使全50 FM上升，均值+.00048606；
U为17/24上升，均值+.00022662，保留7个反向实例。这是输出层代数干预，不是重跑完整低精度网络，
更不是同时替换上游M的完整语言干预。小的末层效应不能代表语言在整条链路无用或解释T/U闭环差额。

科学取舍：降低“需要先给当前Writer补上语言输入/简单语言融合”的优先级，也不能再将其视频功能描述为完全无视目标的模板。
结合§52的真实内容匹配、§59的有限教学适配效应和训练/held差异，下一关键问题仍是**已有条件作用是否在自己的生成状态
和新关系中被正确调用**。已派§23具体回答前一部分；本分析没有证明后一部分、时间必要性或修正有效。
不据交互比例追加层扫描、新language head、任务字典或新正式训练；相关完整理论与历史反例统一在机制§61。

另有与已派生成诊断直接相关的消费者边界（机制§61.5）：FM监督前7维，但官方10步持续更新32维suffix、最后才裁7维。
窄读T/U900、T2340的action_out公共B/Writer O，后25行都为0；MT300同处B后25行也为0。
然而上游条件M仍可改变H，从而改变冻结source输出后25维及下一步action_in的输入。
这解释了为何保存完整32维生成状态，而非把它视为纯显示padding；没有据此认定误差、改变loss或新开干预。
tmp `main_suffix_width.py/json`保存实际参数读取；临时O索引按实际Writer顺序纠正后才采用数值，未影响任何原件/模型。

学习解释同时核到一个必要限定：7a0fdcba冻结`FormalData.event`从排除teacher后的49条中抽28条query episode，
所以实际有限池不是teacher/query严格条件独立。固定teacher v的期望风险为`R_(-v)=(50R−R_v)/49`，
完整均匀teacher轮且固定W时再平均才回到R；有限28query、部分轮次及实际改变中的W不满足该精确等式。
它限制旧理想风险论证如何用于当前训练，不证明某种视频语义、标签矛盾或性能根因，也不触发数据重配；详见机制§61.4。

## 234. T2520回到154：恢复部分先前能力，尚未回到2340高点（2026-09-29）

主讨论直接读新400 JSON、sealed2340/2430/MT及此前六点成功集合，复算逐行scene/video/语言/env与policy RNG配对、
每task50不同视频、得失和阶段谓词；读bank/合同及official/train外部退出。新NPZ/PT内容继承Sol已执行消费者，
不重复全扫或读取未完成节点。脚本/结果为既有tmp `main_2520_analysis.py/json`，0模型/环境/GPU。

2520=154/400、breadth6；task3/6/11/16/23/26/31/39为43/6/45/4/0/34/22/0，suite为49/49/34/22。
相对2430=147：R132/G22/L15、净+7、churn37、Jaccard.78107；相对2340=161：R140/G14/L21、净−7、Jaccard.8；
相对MT153：R115/G39/L38、净+1、Jaccard.59896。固定8task bootstrap区间依次[−1,+5]、[−3,−.5]、[−4.75,+6.25]pp，
继承已核统计消费者，本次没有重抽样。没有任何task的总成功数超过2340，不能把总体回升称为新能力高点。

2430丢失的26个2340成功中，本次恢复13个；此前六点全成功却在2430丢失的8例中恢复5例，
另3例(26,33)、(31,11)、(31,39)仍失败，并新丢稳定例(3,4)。22个相对2430新得中21个在1890–2340曾成功，
只有(31,16)未曾成功。说明这轮主要是恢复和交换，不是22项全新能力，也不是2430之后持续单向遗忘。

task31由17→22，得10失5；10个恢复例在2430的阶段状态中7个仅奶酪曾入篮、2个仅黄油、1个分别曾达成两物。
本次全50例两物各曾达成为23、最终同时成立22；奶酪曾达成后又丢失由7降4。该行为支持局部获取/保持恢复，
但“各曾达成”不等于曾同时成功，也不从阶段谓词跳到神经层根因。task23/39继续为0。

训练2340→2790已经完整exit0：外部6972.142410秒×4卡=7.746824900GPUh，450新更新、50400逻辑query；
五完整ECP/旧2340 metrics前缀与游标的实际CPU读取沿Sol验收继承。2520 official exit0、1.137955947GPUh。
源7a0fdcba、evaluation-only 3cb113c6；2610恢复与2700/2790完整评测仍按原§21收尾，未核其分数或把当前账称最终账。
本次反驳连续下降的简单描述，不证明已恢复高点、达到上限或需要自动2790后追加；余下原节点无需重新派发。

## 235. 视频功能进入真实十步生成，但平均动作误差掩盖了运动改善与开闭选择缺口（2026-09-29）

§23原件位于`/data1/user/ymdai/ember_runs/operator_chain_diagnosis_20260929/sampler_transport`。
主讨论直接读实际脚本/合同/完成和退出，读取40条官方10步生成路径及24组full在β路径上的cross velocity，
结合各条件已存M_out核有限分解与真实消费者；未新增前向、环境、更新或held标签。
独立分析为既有tmp `main_sampler_transport.py/json`、`main_sampler_gripper.py/json`；
endpoint递推最大残差7.15e−7、与执行者报告MSE最大差3.35e−8，不要求逐bit一致。

固定train0/12/20/32各28query、每task两teacher、每query一个相同Gaussian noise；β/MT每task只生成一次。
24full、12自身β、4独立MT300，共40批生成，不是40个环境episode。下表为8条件等权的实际前5归一化动作MSE：

| 模型 | full | 自身β | MT300 | full−β | full−MT |
|---|---:|---:|---:|---:|---:|
| T2340 | .12422832 | .14028243 | .11960994 | −.01605411 | +.00461838 |
| T900 | .12425028 | .13890139 | .11960994 | −.01465111 | +.00464034 |
| U900 | .12562993 | .14053202 | .11960994 | −.01490209 | +.00601999 |

全部24条件相对自身β的前5均改善；全50及unpad结论同向。削弱“条件功能只能在专家动作插值上起效”的解释。
但各条件均值有益不等于每个query有益：T2340/T900/U900相对β改善145/145/157个query×teacher项（各224项），
相对MT改善145/140/137项；同query两teacher及共享β/MT不能当独立样本数。
T2340−T900前5仅−.00002196，而全50为−.00163811；该训练小面板不能解释held148→161全部增长。
U900−T900全50为−.00197634，前5却+.00137965；剔除前5内的重复padding仍保留符号。
这进一步收窄“U拟合更好”的范围，却不证明绑定造成了闭环排名。

**均值不代表所有动作维度同向。** 对T2340，运动前6维平均MSE为.09653835，MT为.10242773；
夹爪一维则.29036813对.22270322，拉高整体7维均值。T900/U900也都是运动6维平均更好、夹爪更差。
task32（开炉灶并放摩卡壶）full相对自身β前5改善.02369480，仍比MT差.02691700；
该task运动6维平均优于MT约.006691，夹爪差+.228567。不能据整体MSE说整条运动生成普遍退化。

沿实际`Pi05LiberoProcessor.unnormalize_action`、官方`_plan_action_chunks/rollout_shard`和安装的
`PandaGripper.format_action`核消费者：source夹爪q01/q99为−1/+1，反归一化后正负驱动开闭，幅值不直接作开闭量。
因此另按真实sign规则读取原预测。在1120个前缀动作位置（两teacher重复查询、含16个padding位置）上，
T2340/T900/U900与专家开闭不同分别81/82/82，MT62；去padding后分母1104，差异数不变。
T2340比MT额外不同的19个位置全部是专家要打开而它仍闭合；其中16个来自task32的两条query、各两teacher。
task32 demo5/frame74：专家前5都开，T两teacher都闭，MT都开；demo14/frame95：专家都开，T都闭，MT后3个开。
这说明差额确有开闭指令分歧，不只是连续幅值惩罚；但它只相对这条专家后续，不证明另一动作必然失败或held根因。

task32的T2340 full与自身β开闭序列完全相同，各teacher均有18/140个不匹配专家的位置。
在这些位置，条件分支实际夹爪改变量均值约.0041，β离零边界约1.0035，未跨过开闭边界，运动预测却明显改善。
全4task也有6个位置被视频纠正、7个被其弄错；不能称视频完全不控制夹爪或一律有害。
这是对“有用的条件响应”仍可能没有纠正关键选择的具体实例，不直接推出新gripper loss、统一放大M或恢复旧端点辅助。

有限路径分解：d=−.1Σ[vF(zB)−vB(zB)]，f=−.1Σ[vF(zF)−vF(zB)]，终点full−β=d+f。
以β误差e为参照，先计d的MSE差为2<e,d>+||d||²，再计f为2<e+d,f>+||f||²；顺序交换会重新分配2<d,f>。
前5先d/后f分别T2340 −.01701282/+.00095871、T900 −.01617698/+.00152586、U900 −.01719210/+.00229000。
三者前5均值的feedback符号在两种顺序下都抵消部分收益，但个例及全50/unpad并非如此；
不能把这叫唯一因果份额、环境反馈或证明去掉反馈会更好。沿实际full轨迹的M_out A h累积也远小于其余终点变化，
支持条件效应不限于action_out的条件投影，但固定轨迹的代数扣除不等于重采样删层策略。完整推理见机制§62。

结论更新：当前有用条件算子可作用于自身真实生成；剩余问题更接近条件响应是否纠正任务所需的对象、关系与阶段选择，
而非所有视频信号在采样时消失。训练小面板也有关键选择缺口，不能把全部不足都推给held泛化。
同时不因发现夹爪局部反例转去解决通用VLA夹爪问题；任何修正须解释视频知识如何改变所需选择，并实际优于旧目标。
旧公共辅助FM改善而闭环下降的反例继续约束后继。本项一次exit0/.139507762GPUh、约925MiB，已释放临时第7卡；
formal§21继续原收尾，本项无新训练/controls/Test/更多采样或自动后继。

## 236. T2790末点159：Spatial/Object恢复而Goal/Long交换，尚未形成窗口净收益（2026-09-29）

主讨论直接读2790新400 JSON及sealed2340/2430/2520/MT，复算完整scene/video/语言/env与policy RNG配对、
每task50视频、成功集合与阶段谓词；直接核bank/合同/worker完成及外部exit0。新NPZ/PT内容读取继承Sol实际消费者，
不复扫捕获或读取2610/2700半批。独立脚本/结果为既有tmp `main_2790_analysis.py/json`。

2790=159/400、breadth6，per-task3/6/11/16/23/26/31/39为46/7/47/7/0/33/19/0，suite53/54/33/19。
对父2340=161：R142/G17/L19、净−2、churn36、Jaccard.79775；对MT153：R115/G44/L38、净+6、churn82、Jaccard.58376。
固定8task bootstrap区间分别[−4,+2.5]与[−4.25625,+8.5]pp，继承已核统计消费者而未重抽样，不称稳健超过MT。
对2520=154为R141/G18/L13、净+5、churn31、Jaccard.81977；这是跨270步比较，不能当作相邻90步结果。

相对2340，task3/6各+1、11/16各+2，Spatial/Object共+6；task26−3、31−5，Goal/Long共−8。
总分接近父点仍伴随具体任务交换，不能解释为所有task都恢复或已经平台收敛；task23/39仍无覆盖。
2790新增17个父点未成功案例中16个在此前已验点曾成功，只有(11,38)在本次读取的1890–2520观察点中未成功。
这不是对所有历史checkpoint的“新能力”判定，2610/2700也尚未消费。

2430丢失的26个父点成功，本次恢复15个。此前1890–2340六点全成功的112例，2790保有108例；
2430丢失的其中8例恢复7例，仅(11,14)仍失，同时另失(26,29)、(26,49)、(31,36)。
因此2430不是已确认不可逆的持续崩塌；但固定队列上仍有交换，不能用总数回升抹去具体损失。
task31两物各曾入篮由2340的24例变20例、最终同时达成由24变19，奶酪曾达成后又丢失由4变5；
该变化包含目标获取和局部保持，不由同一旧行为现象再次宣称定位了新的内部机制。

本节点训练7a0fdcba、evaluation-only 3cb113c6，完整macro2790身份直接核到；official1818.693769秒×2卡，
exit0、1.010385427GPUh，0失败/孤儿沿实际消费者继承。整批仍有2610/2700原队列恢复，末点完成不等于五节点全完成。
Sol回报两旧尝试分别保有300/316行并因严格初始RGB断言退出，原有效行/失败/成本保留；main未读这些中途成绩或另作故障排查。
已退出作业14.187206566GPUh是Sol回报时快照，运行作业另计，不作最终账；仍受18GPUh硬限约束。

科学取舍：已经取得的窗口端点没有比父点更高，且不是持续下降；保留两种事实，完成原剩余节点再解释完整曲线。
不从局部回升自动接下一450步，也不由端点−2宣布全局最高；未选点、未释放controls/Test。
Owner另纠正主讨论：U的均值拟合优势不能外推部署优势早已有判断。§23前缀排序只是实际生成场景的新增证据，
不重复计为新的机制认识，尚未解释绑定的因果作用；后继分析必须实际缩小竞争解释，而非重复报告不同读出排名。

## 237. 2790窗口全收束，Owner直接释放最佳T的Test并明确复用已有基线（2026-09-29）

最后2700=154、2610=160均由main读400 JSON、同2340的scene/video/语言/RNG配对及正式完成；新NPZ/PT继承Sol实际读取。
五点2430/2520/2610/2700/2790为147/154/160/154/159，均breadth6，task23/39仍0。原2340=161保持最高，
这450步没有新高，但后段恢复排除了把首跌直接称为连续崩塌；不把有限窗口当作全局最高证明。
2610对2340为R144/G16/L17、净−1，对MT为R120/G40/L33、净+7；任务簇区间分别[−1.75,+1.5]和[−3.25,+7]pp。
main核最终账的17条退出记录与无活动项记录，累计14.594256020/18GPUh，失败1.787431341；2700实际孤儿NPZ34，纠正Sol此前误报31。
300/316有效行均沿原队列保留补缺，未为得到更好分数重跑；scene单环境恢复通过不证明原故障根因已定位。

Owner随后明确可直接看最佳checkpoint的Test，覆盖先明显胜MT才触发的旧条件；又纠正MT/Source已有结果，应直接复用。
main直接读旧Test两组各400原行：Source75、MT121，基于相同coverage Test8/50init/seed7与官方动作流程。
旧合同无完整scene注册、无capture；本次复用这些基线，不为补完整RGB配对重评800行，也不冒称这种额外证据已存在。
main按完整验证最高/同分早选规则最终冻结T2340/e2afbfd7，唯一新增T Test400；新授权/信息墙/来源/预算见active design§24。
此选择是Owner授权的验证选峰，不能改写为稳健超MT、视频因果或旧资格通过；Test不反哺改选、梯度或设计。

## 238. 固定T2340 Test115：未形成相对MT的整体优势，转入机制定位与修正验证（2026-09-29）

Owner在Test完成前已授权按其总体结果自主选择后继分支，覆盖上节当时的设计反馈暂停；
固定T不重选、Test不进入共享梯度、后续不再称盲测的边界保持。Owner随后再次强调正式重训前须先定位机制/特征原因并验证修正。

main直接读T/MT/Source三组各400正式JSON、合同、T selection/bank/completion和两次外部退出，
复算任务/初态/语言/env及policy RNG共同前缀、50不同teacher/task、所有成功集合及阶段谓词。
实际脚本/结果在既有tmp `operator_chain_diagnosis_20260929/main_selected_test_analysis.py/json`。
新400 NPZ/PT内容验收继承Sol实际消费者；main已读其verify_selected_test.py，不重复全扫capture。
source模型记录、tokenizer、policy口径一致；旧norm临时路径已退休，本次从旧合同记录的clean c9844dfc Git读取同一文件并与新norm内容核同。
旧基线无完整scene/RGB capture，只能确认登记task/init/RNG比较，不能追认为validation式full-scene严格配对或bitwise一致。

| suite/task | T2340 | MT300 | Source1000 | T对MT R/G/L |
|---|---:|---:|---:|---|
| Spatial8，盘旁黑碗放盘 | 10 | 11 | 3 | 1/9/10 |
| Spatial9，柜上黑碗放盘 | 0 | 9 | 0 | 0/0/9 |
| Object0，字母汤入篮 | 1 | 11 | 0 | 0/1/11 |
| Object8，布丁入篮 | 19 | 20 | 0 | 12/7/8 |
| Goal4，碗放柜顶 | 50 | 42 | 36 | 42/8/0 |
| Goal7，开炉灶 | 0 | 4 | 0 | 0/0/4 |
| Long0，字母汤与番茄酱入篮 | 1 | 0 | 1 | 0/1/0 |
| Long3，碗入底抽屉后关闭 | 34 | 24 | 35 | 19/15/5 |
| 合计 | 115 | 121 | 75 | 74/41/47 |

breadth依次6/7/4；四suite T为10/20/50/35，MT为20/31/46/24。
T−MT净−6（−1.5pp）、churn88、Jaccard.45679；固定8task簇95%区间[−10.5,+8]pp。
T−Source净+40（+10pp）、R60/G55/L15、churn70、Jaccard.46154，区间[+1,+20.75]pp。
区间沿已核实际消费者继承，不含重新训练不确定性；总分接近不等于策略等效，也不能说已证明T显著更差。
T在MT与Source均失败的22个案例成功，MT在另两者均失败的42个案例成功；这不是融合/union部署建议。

**哪些解释得到收缩。** Validation161对MT153的优势没有在当前Test重现，说明尚无稳定超过强MT的广泛迁移证据。
不能用115对161的跨任务差直接诊断过拟合、统一难度或整体失效；Test任务不同且Validation曾选峰。
当前差额是Spatial/Object共−21与Goal/Long共+15抵消，不能由−6声称全任务几乎一致；
Spatial8两者10/11却只共享1次成功尤其说明成功数相近掩盖交换，完整scene缺证的界限仍保留。
T在Goal4取得50/50并保留MT42，Object8也比Source多19个成功；这些正例不能因Test未胜MT抹去，
但Test未运行公共β或视频干预，不能单凭本表把这些增益归因于教学内容。

**完整任务的不同缺口。** T的Long0番茄酱44/50曾入篮且最终保持，字母汤只有1/50；
Object0的单物字母汤也只有1/50。因而不能把Long0的全部失败归于多阶段顺序、末段丢物或递归记忆遗忘。
Long3则50/50碗已入抽屉，34/50关好；其16个失败属于已放入后的关抽屉未完成，而非都没学会搬碗。
对应初态谓词均为false，以上不是把初始满足当后续获取；没有旧MT/Source轨迹，不能比较三者具体阶段的因果差异。
这些是行为接口定位，不证明相同物体在不同场景里具有同一内部失败，更不把它们直接变成新loss或数据选择。

科学决定：按Owner事前分支进入深入机制分析和有界修正验证，暂不运行one-shot/task-local RL或更多原样续训。
Sol已收到§25，继续固定2340的四个validation视频/公共β对照，不需重复派发；它们不会反向改选或把shuffle损伤转成架构依据。
主讨论须在non-held数据上连接真实H/X→K/Value→M→自身Ah→控制功能，检验具体竞争解释后再作修正训练；
现有训练FM/实际采样正例排除了部分“完全无作用”说法，却没有确定当前性能缺口的唯一根因。

T训练源e2afbfd7，evaluation-only6fb423b3 clean detached；bank400条件/401因子，18worker exit0，0 orphan沿实际验收继承。
bank1133.4067秒/.314835193GPUh，official665.3960秒×6卡/1.108993398GPUh，总1.423828591/3GPUh。
唯一原件root=`/data1/user/ymdai/ember_runs/operator_selected_validation_20260929`，报告`analysis/selected_test_readback.json`，
新正式原行位于`test/T/evaluation/correct400`；MT/Source保持原data0基线，不新跑800行。

## 239. T2340完整视频对照：有益条件作用与跨视频保持必须分开（2026-09-29）

§25四组完成。主讨论直接读取新旧各400原行、bank/contract/completion及外部退出账，复算配对集合，
核同一2340/e2来源、f207a12e评测、same-task-other确实复用全部旧因子，检查wrong/shuffled的实际完整编译消费者。
新1600 NPZ/PT的内容验收继承已读过的`verify_selected_control.py`与Sol逐行原件结果，不重复全扫。

| arm | 成功/400 | breadth | arm−correct R/G/L | task簇95%差额区间pp |
| --- | ---: | ---: | --- | --- |
| correct，封存参考 | 161 | 6 | — | — |
| same-task-other | 150 | 6 | 131/19/30 | [−5.25,−.50] |
| cross-suite-wrong | 65 | 6 | 48/17/113 | [−47.50,−5.00] |
| shuffled | 1 | 1 | 1/0/160 | [−65.75,−14.75] |
| 自身公共β | 103 | 5 | 79/24/82 | [−35.25,0] |

逐任务按3/6/11/16/23/26/31/39：correct=45/6/45/5/0/36/24/0，other=44/6/43/5/0/33/19/0，
wrong=3/2/11/2/0/32/15/0，β=6/3/30/0/0/41/23/0；shuffle只在task3成功1。
所有区间只包含固定任务/场景的重抽样，不包含重新训练、检查多checkpoint选峰或其它视频映射的不确定性。

**三种效应不能混为“视频理解”。** 正确相对错误的净96条差额可精确记作 `(161−103)−(65−103)=58−(−38)`：
既有正确条件对共同β的净改善，也有错误视频对β的破坏。113条correct成功/wrong失败中，73条β也失败、40条β成功；
不是全部96条都来自新操作能力。另一方面，correct与other都成功而β失败的64例，说明条件收益不全依赖选中的单一映射。
这仍非独立language baseline或Test视频必要性；cross-suite同时换视觉分布，shuffled只显示冻结方法的强顺序/分布敏感性，
不证明理解正确操作次序，不反哺架构/训练。先前训练4task功能优势现在有validation闭环内容响应补证，不能称从未知道视频有功能。

**收益集中且有共同损伤。** 两种正确视频同时胜β的64例中，task3/11贡献37/16；两种视频同时丢失β成功的14例中，
task26/31贡献8/4。task3正确/other保持42个共同成功，task11保持43个；task31却只有9个共同成功（24与19总分），
因此同task视频稳健性的问题主要是某些任务上的状态×视频交换，不能说所有任务同样不稳定或简单挑到劣质视频。
两种映射都完整使用相同50条视频，没有“选好视频”的操作。other比MT153低3；correct高8随映射改变未保持。
同时胜MT的27例和同时输MT的26例均保留，不能用161单映射宣称普遍优势，也不抹去局部超越。

**行为约束。** task31正确/other/β曾达奶酪目标为38/40/44，曾达黄油目标为29/21/25，最终两物成功24/19/23。
other有11例奶酪曾达成而最终丢失，correct与β均为4；部分是获取不足，部分是保持不足，不能统一解释成第二阶段或记忆尾部覆盖。
task23所有臂均未达目标；task39正确/other/wrong均两目标从未达成，β仅一例曾达第一目标。不能把零分都归给视频支路。

**实际算子的额外否证。** main只用§22保存的T2340八个正确train视频H/X与真实权重，复算Q8/V8/out的写入分解。
第一段写入到最终M的范数保留范围分别.0405–.1864/.1107–.2787/.2868–.6165；这不是语义保持率。
末层固定自身输入、仅将递归M替为相同K/V的无擦除累积，完整50 FM在8/8变差；匹配原M范数后仍6/8变差。
task32两teacher的前5总体更差而夹爪分量略好，不能用局部夹爪改善掩盖运动损伤。
这降低“衰减明显→取消擦除即可修复”的支持度；仅三个target、单次FM和末层有限干预，未排除上游或经重训的所有记忆方案。
没有由此放行gate/scale/decay搜索。实际数字与链路解释见机制§63。

完整账4.590690977/6GPUh，6个GPU job；wrong外部exit124发生在400行和6worker exit0后、聚合前，
同f207 freeze的CPU aggregate exit0，无episode重评，超时费用完整保留。Test旧1.423828591另列。
原件仍在`operator_selected_validation_20260929/analysis/selected_controls_summary.json`及四个canonical臂；
主讨论过程在既有tmp `main_selected_controls_analysis.py/json`、`main_write_survival.py/json`。
尚无已验证修正、重新训练或RL/one-shot结果；后继只认当前progress与新具体合同。

附加实际线索：公共Q8/V8 A在2340→2790仍有约94%元素相同，而它们与Adam m/v真实存为BF16，Value C/O为FP32。
这不等于零梯度或卡死，也不能解释所有平台；输入/输出A仍在变，输出25/32无直接信用是合法缺路。
§26在本就需要的四次临时单步内加同梯度/同状态的CPU算术影子，核实有益小位移是否被存储舍入抹掉，
不据dtype直接切换训练或搞精度扫描。细节与近邻边界见机制§63.4。

## 240. 独立episode有限更新与结构投影：未支持前缀目标、精度修复或简单扩宽出口（2026-09-29）

§26完整结束。main读实际脚本的A/B分区、真实76因子余切/native重放、同父独立Adam与FP32影子，
读完整readback/precision/两退出回执并复算聚合；另直接复算四个task各一个父B原始预测风险。
48路径的全量PT读回继承Sol已审消费者，不重复全扫或重跑模型。e2afbfd7真实T2340/绝对2341，
两个teacher组各F(完整50 FM)/P(前5 FM)一次，共四个独立临时更新；不是0更新或正式checkpoint。
A28与B20、两teacher恰好组成每task50个互斥episode；全部task为训练0/12/20/32，无held/环境。

P使自身A前5 FM在8/8下降，但在独立B上优于F仅5/16，另一teacher仅2/8。
B前5平均风险变化F −2.68174e−5、P −1.21701e−5；另一teacher为+1.51e−6/−1.12e−6，符号随task交换。
P全50/有效future平均反而+2.40e−5/+3.56e−5；两臂前5运动误差略升、夹爪误差略降。
裸LoRA的A50/B点积同/跨teacher分别都只有2/8正，真实Writer重放后仍2/8；
没有“裸信用普遍有益，只在J处系统反转”的清晰模式。实际一阶预测平均为正而有限风险差平均为负，
该量级的线性余项不能忽略，不把它命名为精确下降方向或单一非线性根因。

**存储舍入是实测事实，有益更新被阻断仍未成立。** 四次clip前norm约.03583/.09779/.04119/.10339，clip1未触发。
同m/v、同实际梯度的FP32拟议位移，公共A有94.79%–95.36%非零元素在转回BF16时丢失，
对应拟议平方位移能量92.45%–93.45%；非零梯度掩码仍成立。B0和Q8/V8明细在原件，Value FP32无该现象。
但影子位移在B梯度上的平均投影F/P为+3.41e−5/+6.96e−5，比实际+8.79e−6/+3.10e−5更正，
不是被抹掉的更新更有利的证据；影子权重未进入模型，不能反过来断言FP32真实学习一定有害。
本项没有给前5目标、精度改训、额外步数或正式重训提供充分依据，按原范围收束。

**main新增CPU架构分析。** 复用这批余切与旧八条T2340实际M，仅看事前已有Q8/V8/out三个位置。
Q8的O为2048×256、列满秩；固定O时其列空间平均覆盖约65.0%的B因子余切能量。
但不能把其余35%直接说成现有架构不能表达：在固定A、允许各视频F任意变化的局部上界中，
计入原有公共B0与共享O的变化后，覆盖约98.45%；未覆盖部分的A50/B点积为负，未显示扩宽出口会释放有益训练方向。
V8的256×256 O满秩；out的O秩7覆盖约99.81%的此处生成风险余切能量。
余下25输出维在完整十步生成中可间接影响未来latent，不能把生成风险余切与单次七维FM的直接信用混淆。

另在Q8当前row(A)之外、每个col(B_v)可见的有效权重梯度分量中，共用A的局部变动覆盖约27.7%。
这是另一种坐标/约束，不与前述98.45%相加或对比为同一容量指标；被共用限制挡住的完整FM/B点积仍为负，
前5项部分为正却不足以改变§26的有限阴性。没有据此采用逐视频自由A或更大rank。
上述是三个位置的局部投影上界，不证明完整Writer可学性或不存在容量不足；实际F仍受视频网络限制。
数学与后继完整选择见机制§64，过程在既有tmp `main_credit_architecture.py/json`，无新GPU/模型/更新。

§26两作业均exit0，外部总.2341999299/2GPUh；新增10.397GiB，低于12GiB合同。
canonical原件在`operator_chain_diagnosis_20260929/functional_credit_transport/`；没有正式模型或新闭环分数。

## 241. 回读与历史相似设计的整体复核，以及父数值消费者的配对限制（2026-09-29）

Owner要求核对旧回读并完整解释当前架构。main复读机制§20、Horizon/Unified设计与相关真实Git消费者，
并核当前model/native/run/credit/data/flow：过去已有外部过程→视觉重读、跨帧Z/H写回native继续计算，
也已完整提出临时LoRA→native回读→新参数的图。最后一种未正式运行；Unified有90/109/67/84的历史阴性；
20260926 NativeConditionalReader只工程完成、正式学习撤回，不能把这些写成同一种已失败实验。
本次冻结T重读只是窄假设检验，不因名字、已有效父模型或额外一遍计算获得收益保证。

原§27两成功组共32新B路径，0更新/环境/held，含一次核验失败共.1202047007GPUh。
真实script保持原公共A/B0，M0只用于native，最终B0+M1，没有叠加M0或新模块。
main核实际调用/完整聚合与四个代表原预测，其余内容验收继承Sol实际CPU消费者。
报告前5均值joint_self/address_input_self/value_context_self/joint_other相对旧父分别
+.00514151/+.00231369/+.00194500/+.00524928；self与other差−.00010777且八条件正负各4。
这些风险仍限固定train B20/单noise，不是closed-loop。

main发现必须说明的比较限制：§26旧父microbatch5，新候选10；首次batch2父核验已有前5差.00078012。
重试核验改5得到0只说明5→5，不能证明10→5的候选差额来自回读。
已沿原预算限定补8父同microbatch10，保留原32/β/MT、不铺数值矩阵；详见design§27.6。
匹配前先限制差额因果解释；§27.6现已完成：父10对旧5平均漂移−.0000690734。
同batch10的joint/X/H/other前5差+.00521058/+.00238276/+.00201408/+.00531835，改善条件1/1/1/2。
仅H链task20/38的微小差翻转，主判断不变；H链全50小改善但前5更差，保留全部范围。
main核新消费者/全聚合，直接复算两个代表父预测；总.1406927840GPUh/约1.3GiB，本批完整收束。

完整理论修订见机制§65：原T已把整视频递归写成条件反馈算子，回读改变的是整视频参数再进入逐帧native的位置。
第二遍仍在M0下读取而最后执行M1，既未保证参数一致，也没有纠正/风险下降含义；
即便迭代收缩也不保证动作风险改善。未来若联合训练须真的穿过两遍native和M0求导，原单遍replay不自动提供这条边。
当前没有新架构学习或正式重训；T已有功能与迁移不足均保留，不以局部阴性整体换架构。
Owner随后明确停止新实验，main已接受，当前不派任何后继；原件入口见机制§65.5。

## 242. 恢复研究后的实际教学调用分解：不支持简单加强记忆（2026-09-29）

Owner新授权覆盖旧停止令。新执行者只读T2340四train任务/两teacher现存原件，0 GPU前向/更新/环境/新held。
真实action_out十步Aq允许精确展开逐教学时刻的七维条件速度，M重建最大相对误差4.88e−7；
main直接读脚本、报告及实际消费者索引，区分FM query_x、A28真实生成Aq及没有Aq的独立B20。

task32两已知开爪反例下，四个teacher×query的末层直接项都很小且净向闭；已保留开方向负项仅.0013–.0023，
不足以解释如何跨越约1的符号间隔。不同转移存在正负抵消；task0/12直接夹爪方向还与完整策略变化相反。
不支持单纯保留早期写入或放大末层残差，不能从这份固定full-path hidden分解定位整个38目标的唯一原因。
完整正反数字、解释及原件见机制§66.1、run root `operator_chain_diagnosis_20260929/functional_address_evidence/`。

main随后只用已存全38余切筛查公共输出空间/读图前目标两项候选，未找到足够投入依据；不是有限干预否证。
按Q/V/动作投影的现存功能信用出现可区分的分组差异，具体假设与结果分支登记在机制§66.3及design§28。
目前未找到经过验证的架构修复，不启动正式重训；真实执行状态以progress顶部为准。

## 243. 完整分支删减无直接收益；冻结共适配不能替代架构学习比较（2026-09-29）

§28固定T2340、四train任务/两teacher/独立B20、同microbatch10父，共24条新十步生成，
0更新/环境/新held，.032783507GPUh，两组exit0。main读真实装载/生成/读回代码与完整聚合，
直接从task12删Q、task0/32删V共六份候选及其父原预测重算前5/全50/运动/夹爪/符号差额。
全局删Q/V/IO前5均值分别+.017060/+.003066/+.000219；task12删Q两teacher各16/20 query变差。
task0删V两teacher前5−.006387/−.001832，但全50和有效future变差，前5收益集中在共同两query，
其余18个query描述性均值+.006789；其它三task全部变差。删V/IO没有前5夹爪符号翻转。
这不支持现成T的全局直接删减；不能据局部V阴性称其无用，也没有解决task32反例或绝对闭环不足。
原件入口及完整取舍见机制§67；实现e0ef50e4保留Git/冻结树，结束的专用入口不留活动main。

Owner随后指出成熟模型已经耦合，大部分冻结改动变差本就是合理预期。main承认此前虽写外推边界，
实际仍过多用冻结即刻收益筛选需要重新共同学习的架构。`R(新结构,旧参数)`与
`R(旧结构,旧参数)`的差，不等于两结构匹配学习后的能力差；单个Adam方向也不能替代后者。
Q/V结果保留为当前功能依赖；回读/输出空间局部阴性保留为未经适配的实际效应，不作新学习关系的硬否决。
后续须从具体特征—算子—信用假设提出有界匹配学习，并保留闭环、相邻保持和迁移判据；
不因纠正而抹去完整历史阴性或自动重开所有旧路线。当前尚无有效修正，正式重训仍未放行。

## 244. 理论与轻量验证收敛：变化增量和逐帧绝对拟合的时钟不一致（2026-09-30）

Owner要求约一小时内用纯数学推导＋轻量机制验证形成主判断，撤回§29 fresh270/correct400长验证。
§29实际0 GPUh/0更新/无run root，只保留afb0b357候选代码及CPU工程检查；未集成main或启动新训练。
本轮执行者CPU三A初值审查、main真实特征代数/既有余切分析全部完成，0 GPU/native或完整policy前向/更新/环境/新held。
完整推导、全部反例与可复算入口见机制§69及`docs/analyses/operator_clock_evidence_20260930.json`。

- T2340的Q8/V8/out A相对identity变化.399/.289/.398，固定成熟X的key角中位48.47°/38.73°/29.30°；
  主讨论直接审脚本/原件摘要，反驳“A从初值基本未学动”。后期BF16丢位不能由此宣布全程地址冻结或主要根因。
- 实际V对Δh线性，递归却等价逐帧拟合`M K≈V`的一步更新。静止V=0仍擦除；固定地址正特征值λ方向的局部例，
  同一总变化L分N步时`m_N=(b L/N)[1−(1−λ)^N]/λ→0`。这是明确算子限制及局部模型，不是完整policy性能定理。
- 首选最小候选保留T实际A绑定/β/38处单LoRA/真实FM，仅把擦除改为`M K diag(1−exp(−RMS(Δh)))`。
  插入相同帧不变、旧M传播非扩张，连续小步写入和覆盖都按特征路径推进；有限stride5不保证任意真实重采样严格不变。
- 在真实H与AX的分段线性路径上细分并重算原Value/key，固定原自身生成Aq评价线性target作用。
  4份→8份的相对作用变化中位，原T Q8/V8/out=.471/.425/.350，新clock=.090/.050/.067。
  1份→8份23/24 site条件更稳定，task32/teacher43 V8仍更差；同平均强度的固定弱覆盖在Q8/V8未取得相同稳定性。
  插值特征不保证有对应RGB，固定query响应不是完整policy风险或成功率，不把这些数当涨分。
- 新clock让独立B余切给更多早期Value信用，却未普遍改善FM/B在O方向的一致性，三个site平均余弦均更差；
  task32 Q8早期比例几乎不增。它降低“更长记忆就是更多正确知识”的支持，不以形式保持性盖过不利结果。

主讨论据此将“变化型Value与逐帧绝对拟合的时钟不一致”列为目前最有明确数学约束、可作单一自洽修正的结构假设。
这不是“已确认绝对成绩主要受它限制”，也不是“已有被擦掉的开爪解”；T/U、LocalField/ProcessPullback、公共辅助、
冻结回读与分支删减的历史边界保留。当前候选值得优先讨论，但未验证性能修复或取得正式重训资格；无自动后继实验。


## 245. 综合限制审查：局部结构缺点尚不能指定性能主因（2026-09-30）

Owner明确轻量可含GPU/学习、不限制参数或修改幅度，B0回读只是例子，实际原因可有多个。
主讨论修正§244优先级：change_clock有数学结构性质，但无实际性能修复证据，不因最小修改/已有实现成为首选根因。
机制§70给出`M=OΣ f_tj k_tj^T P_t/50`及自身`M A q`的完整联系，区分特征内容、关系写入、执行调用与共享学习。
原FM已包含生成LoRA改变自身hidden的真实信用；缺少本次M参与教学读取是计算限制候选，不能称反传断路。

仅CPU读取§26八条件已保存余切，固定β/A/native的152个P/C/D/O梯度：同task两teacher A余弦.897/.889/.793/.917，
跨task均值A余弦范围−.0396到.0393，不支持全局强负冲突。A→同task B（十步前5风险）余弦为
.0355/−.1692/−.0723/.0699；直接B余切仍为−.1430/−.1182/−.0332/.0207。
解除task共享也不能使task12/20获得普遍有利局部方向；Adam预条件和有限更新仍须实测，不能由此定优化根因。
逐项结果在`docs/analyses/operator_learning_credit_evidence_20260930.json`，CPU脚本在既有main tmp。

拟定§30同起点S/P/D短学习用于区分有限形成/保持，但首消费者加载/NFS阻塞、未完成GPU前向。
Owner随后明确不等待，main即时Steer撤回全部准备/重试/计算。没有新增学习科学结果；
私有Writer即使阳性也不能证明视频语义充分，因私有路由可绕过共享模型需要完成的条件判别与迁移。
当前没有唯一根因或已验证有效修复，§29/30都不能自动恢复。

## 246. 撤回单批不终止研究：B0路径、teacher方差及缓存残差学习（2026-09-30）

Owner“那为啥就停下来了？”纠正main将撤回§30扩大为整体停工的误读。main继续原件分析与轻量CPU闭式拟合，
没有恢复§29/30，没有新模型/GPU/环境/held前向或正式权重更新。完整论证见机制§71。

从原§26全38处B余切G及公共B0总梯度g估计native间接项r=g−G；Q/V包含BF16累计舍入，
action_out FP32且无native反馈的零项得到实际核对。A50的`||r||/||G||=.0607–.1751`、
余弦`−.1086–.0308`，没有普遍严重抵消；不据此拆公共B0，也不外推到尚未拆分的A。

原B20相同query/noise下两teacher的平方误差可精确分成平均预测误差与两者离散项。
后者仅占四task前5总MSE的`.14%–.38%`；task32夹爪离散项`.0001903`，只占相对MT额外`.026205`的约`.73%`。
两teacher主要共同犯错。这不估计全部teacher分布，也不否定held闭环的成功集合交换。

保留父T并添加零输出末层残差`Γ Z_read A q`，匹配7×256出口，比较原β-native与完整T回读。
事前固定ridge，无调参；四次留一train任务及两折14/14跨episode。训练拟合下降，测试均恶化：
跨episode全50变化β/self=`+.010341/+.010963`，留任务=`+.169709/+.276483`。
主计算2.76秒CPU，原M重建相对误差最大4.15e−7；它是固定末层单FM代数，不是10步生成/完整两遍联合学习。
后补直接B控制也不能迁移：共享跨episode`+.034211`、task私有`+.117414`，故该小拟合不具备唯一归因Writer/回读的分辨力。
不把可拟合性当迁移，不为阴性再扫正则/步数；当前仍未找到已验证有效修复。
结果已合并于`docs/analyses/operator_learning_credit_evidence_20260930.json`，逐层B0完整版与全部脚本在既有main tmp。

## 247. 实际Value可达性不支持普遍地址失联；转入有界共同学习检验（2026-09-30）

对T2340八train条件三site，用真实K及其未来覆盖构造`T_v=concat(P_t^T K_t/50)`，
精确关系为`M=V_stack T_v^T`。重建原M最大相对误差1.19e−6，CPU约2.32秒，无模型/环境/梯度/optimizer执行。
S=T_v T_v^T的谱很宽，但小于最大特征值千分之一的方向只承载真实A28十步前5执行特征能量：
Q8 3.7%–5.7%、V8 4.5%–7.3%、out 1.7%–5.4%；独立B20自身B余切对应能量分别2.4%–6.6%、8.0%–12.7%、3.4%–8.4%。
task32两个开爪反例输出层弱方向占1.3%–1.8%，并非主要写不到这些实际执行方向。
任意Value的宽松可达性不是共享Value网络可学性，不据此宣布地址完好或语义已学会，也不据条件数大做白化/RLS。
详见机制§72和既有learning_credit JSON的value_reachability；A28/独立B20消费者始终分开。

这项结果与旧输出空间/B0拆分一起，降低继续寻找普遍线性硬阻断的优先级。
主讨论选择已有结构证据的变化驱动覆盖做一次fresh共同学习；保留旧功能反例，未称其为绝对性能主因。
Owner夜间新授权允许确定修正后自主启动训练；新design§31只到候选270及一次correct400/固定A28，
不重评T/MT，不自动长训或恢复§30。实际范围/进程只看progress，待结果再裁决。

## 248. 变化驱动覆盖共同学习270：尾段信用集中缓解，固定FM收益小且不均匀（2026-09-30）

§31候选完成fresh270，训练517/读取360。main直接读取既存8full+4β的A28面板，并与旧T270同teacher/query/noise比较；
0新增GPU/native前向/更新/环境。CPU代数约4.35秒，48处M重建最大相对误差7.87e−7，信用展开差≤4.86e−11。
Q8/V8/out首写Value梯度传播比均值分别由.03545/.12471/.19954提高到.18209/.52661/.82993，24处均增；
Q8尾三次占绝对局部缩放信用由.79270降至.15020。原始Value幅度均下降，最终M尺度混合变化，不能概括为单纯放大。

完整前5 FM均值改善−.002376（1.73%），β均值反而+.000156；条件增量因此改善−.002532。
train0/12/20/32的前5差分别−.001321/+.004718/−.007787/−.005115；task12夹爪FM额外+.016470。
task20两teacher的全50视频条件作用由略坏转好，task32也在β略坏时改善；task0则主要随β改善。
较早Q8有符号局部信用多数有利，但task20末层M缩放导数为正，不能由末层或信用平均化归因完整收益。

见机制§73及`docs/analyses/operator_change_clock_270_evidence_20260930.json`。
该结果加强“候选确实改变了实际学习联系”，尚未确证绝对性能主因；correct400仍在执行，不能据固定FM提前续训/选点。

## 249. 变化驱动覆盖270完整闭环与后继取舍（2026-09-30）

§31候选124/400，四suite25/42/36/21、breadth5/8。main直接配对全部原行：对原T270116保留89/得35/失27，
净+8；对同scene MT153保留96/得28/失57，净−29；对成熟T2340161净−37（训练预算不等，仅背景）。
task3/6/11/16/23/26/31/39为21/4/42/0/0/36/21/0；收益集中3:+12、11:+11，损失31:−11、6:−5。
完整allocation8.443905GPUh、输出约11GiB，270 ECP/bank/A28/correct400完成，没有把工程失败当科学阴性。

新读既有轨迹：task3目标几乎未动39→24/50，其中另一碗抬起18→4；14新增成功中13个来自旧目标未动场景。
task31黄油曾入篮32→22，15丢失行中13个仍曾完成奶酪却从未完成黄油；主要不是两物都曾完成后失去最终判据。
信用传播改善是真实的，且有具体目标操作收益，但未形成广泛能力恢复；降低“早期覆盖是主要瓶颈”的支持度。
原U270也有较分散信用及相似分布，后期仍退化；原T450也有类似任务交换。不能把信用均匀化当完整方法修复。

候选最后90步的同事件在线FM在35/36训练task较低，整体仍在变化；270只覆盖每task30条teacher、前150步warmup。
main选择一次有限450补点完成首轮50条teacher覆盖，见design§32，不直接长训到900/2340、不更换架构/重评参照。
它检验真实能力能否扩大并改善保持，不以在线loss选模型；若仍只有原有交换而没有有意义的强MT以上能力，
不再仅以学习可能更久为理由延伸clock-only主修复。机制§74保留完整解释、竞争假设、历史与不确定性。

## 250. 变化驱动覆盖450仍123/400，关闭其主要修复解释的继续投入（2026-09-30）

主讨论直接重算完整原行：C450123、C270124、原T450122、U450143、同scene MT153。
C450对C270保留100/得23/失24，churn47；对原T450净+1、对U−20、对MT−30，breadth仍5/8。
候选任务3/6/11/16/23/26/31/39为17/3/46/0/0/37/20/0。相邻保持80.6%高于原T同期68.1%，
但新增23也少于T43/U37，整体没有增长，不能把少交换称为稳定性修复。原T450的7/8含两个仅1例成功task。
C270相对T270的35新增中26个保留到450；Object真实收益继续存在，Spatial原收益保持较弱。

既有轨迹：C270→450目标碗几乎未动24→32/50，丢失的7个task3旧成功全部变成目标几乎未动，只有1个另抬其它碗。
Long新增9个都来自原只完成奶酪，丢失10个里7个又只完成奶酪；第二目标的获得仍交换，非部署中继续擦除编译记忆。
首段信用确实改善（270原面板），完整能力未恢复；按预登记停止clock-only900/2340延伸及gate/scale/seed/LR小扫。
不否定整个T，也不称450已证明收敛上限。实际剩余限制仍是特征—条件作用—独立状态的共同学习联系，唯一根因未定。

正式批5.176389GPUh；Owner工程复现/取证随访.020623单列，总5.197012未越6；§31费用8.443905单列。
原件root `operator_change_clock_continuation450_20260930`，主讨论分析见机制§75及配套450 evidence JSON。
Owner最新允许必要分析实验并要求论证完成后暂停讨论。新design§33以同T2340起点、共享/任务私有Writer/直接B，
使用新抽训练帧、独立于本次更新的B episode和有限闭环，检验可学性与共享约束；它不是新架构资格或旧§30自动复活。

## 251. S/P均20、直接B23：现有图可修正，主要学习限制仍未唯一定位（2026-09-30）

design§33完成，父T2340固定A/B₀/native，64步每条件1792查询；四task×两teacher×四init，
每臂32但只有16个不同物理初态。main直接重算128原行与24份末点预测，见机制§76及远程审阅目录小型证据。
父17、共享S20（R16/G4/L1）、task私有P20（R15/G5/L2）、condition私有D23（R16/G7/L1）。
task0/12/20/32分别父8/3/3/3、S8/4/3/5、P8/4/4/4、D8/4/5/6；task0天花板不算修复。
P对S得4/失4，D对S得5/失2、对P得6/失3，不是成功集合包含关系。
task32 teacher17的D补齐三例“开炉后未放壶”，到4/4；teacher43却未取得S/P修好的init3。
这支持有限真实可修正、弱化单纯task共享解释，尚未把D优势归到Writer某一模块。

固定A-FM由.11077084到S/P/D .10897617/.10827229/.10795090；B20前5 MSE各改善约.00819/.00836/.00742，
full50/valid-future均变差，主要前5改变量来自gripper；离线排序没有解释D闭环较高。
B只独立于这64步新增更新，父T已训练过这些train数据；P/D均不是未见task部署方法或理论上界。

只读已存Z与D实际修正的CPU最小二乘：初始Z的共享O可表达95.86%诱导BA参数能量（B空间81.64%），
保留弱方向截断依赖、site差异及剩余方向可能有功能重要性的边界。
S/P末点共享拟合比例虽到99.17%/99.45%，但绝对残差2.6313→2.6267/2.6628几乎不变，
是目标能量分母扩大，不能称Z学习修好了剩余限制。无拟合后模型评测、无新GPU或更新。
该结果降低巨大输出不可达解释；不把几何能量百分比当闭环能力或建议白化/求逆。

当前优先解释从合法视频特征到可跨状态调用、共同学稳的条件作用；单纯覆盖时钟、解除task共享、
扩大输出空间、删除B₀的充分性均缺支持。Value的局部依赖和实际非线性消费者仍是条件性理论问题，
完整时序上下文的历史负例保留；没有已验证主因或解决方案。
§33成本2.100774/4GPUh，全部退出，按Owner要求暂停新增实验并讨论。
已准备仅远程Git可读的专家数学审阅入口/完整曲线/逐例证据/提示词，不预设新模块，不联系外部专家或自动派发训练。

## 252. 公共/条件功能重审与PZ21：当前修正可部分表达，共享获得与迁移尚未解决（2026-09-30）

Owner授权认真评估A/B₀公共基础、M视频适配、完整架构及损失，main核实际源码、训练合同、最近似完整历史，
并完成一项预登记投影消费者和两项已有原件CPU分析；完整论证见机制§77，三个小型evidence JSON均在docs/analyses。
公共基础＋视频定向适配是合理的功能原则，继承T的共享读写；A仍兼任实际地址，参数命名不等于知识被纯净分开。
当前full FM已等价要求`δ=f_full−fβ`修正`rβ=y−fβ`，重命名残差loss没有新增信用。
它没有独立要求β成为强公共策略；需明确该功能目标，但旧公共aux148/control158不支持直接重启同一辅助配方。
公共风险与完整风险也不是数学上必然冲突，平方和重写中的δ项不能被单独当作直接M正则或冲突证据。

已有B20四train tasks/80不同专家查询/两teacher上，前5风险β/MT/T为.142245/.128359/.117854，
而旧A28的T/MT为.124228/.119610，排序相反。M有真实补偿能力，有限离线面板不能据此选出最终策略或定位全部缺口。
M变化与MT对β变化的相似性只是函数描述，不是公共/视频知识的因果比例；有限池跨episode也不自动保证视频学到新任务规则。

§34唯一PZ固定父Z/A，将已知D修正投影到八条件共享δO，BA覆盖95.86%；实际32行父17/S20/P20/D23/PZ21。
main重算五臂全部原行及八份PZ/D/parent预测。PZ对父R16/G5/L1，对D R19/G2/L4，对S/P均R17/G4/L3；
task0/12/20/32为8/3/4/6，仅16个不同物理初态。
PZ保留task32两例D新增放壶，但teacher17/init3失去D放壶、teacher43/init3反而新增；task20/12也有具体得失。
前5修正与D均值cosine .991683、相对L2误差.143274，PZ−D预测MSE .00012650；相近参数/动作没有保持全部成功集合。
降低“有效修正主要在现有固定Z/O形式之外”的支持，没有证明同一共享Writer已学会生成它或未见task能迁移。

另用同一1e−6截断在已有张量上做两个teacher互换拟合及四次留一train task拟合，CPU3.99秒，无新模型/adapter。
拟合部分解释97.65%/96.92%，排除条件上38/38 targets误差均超过零修正，逐层误差比中位12.39/33.33。
这限制全条件高覆盖的可预测性外推；无正则弱方向拟合及非唯一D标签的边界保留，不把普通插值失效当EMBER特有根因。
不扫截断/正则或据此加诊断链。

当前有依据保留β-native、实际A绑定、38处单LoRA及跨episode真实FM；局部Value与递推仅是有用候选实现。
完整上下文调制动态Value且保持A寻址，是更值得有限检验的候选，尚无新架构收益或实际语义混同证据，LocalField反例不清零。
不同时改自由query、递推、公共课程和多项loss；未启动新训练，未确认唯一主因或完整修复。
§34实际.30101382/2GPUh、峰值新增.85770/8GiB，代码511791cb，全部消费者退出；原件root
`operator_projected_repair_consumers_20260930`，本轮研究完成并交付Owner讨论，没有已派后继。

## 253. 后期公共辅助不等同fresh共同学习：修正外推并隔离目标干预（2026-09-30）

Owner追问后重新核清：旧aux是T1800、原Adam/floor LR1e−5继续90步；固定四train公共FM改善，完整correct148/control158。
没有测公共闭环，也没有从初始化训练双目标。该反例限制“公共拟合改善自动带来完整控制收益”，
不能未经检验否定从初始化建立两个函数的分工；主讨论撤回§77中赋予它过强的fresh否决权重。
Owner随后授权自主推导、实验和新训练，不再等待批准；仍须机制/历史与渐进预算，不能机械照搬Owner或专家候选。

机制§78从实际`fβ`与`fβ,φ`推导：full-only只约束合成函数；公共风险使β误差即使可由M补偿也付出独立代价。
在独立任意函数的理想模型下分别对应`E[y|q]`、`E[y|q,V]`，当前共享A/native/有限优化不保证同时实现。
现有真实反传已完整；新项只加公共76因子信用，P/C/D/O仍只接full；不能将平方和重写误当直接M正则。
fresh首步B0梯度加倍不等于Adam位移加倍；关键是此后一直共同学习，而非已共同适配后的短窗重分配。
理想固定task的条件独立还容许视频增量为零，故更强公共目标可能减少M学习需求，并非自动获得视频知识。

active design§35只改目标：原T图、原初始化/事件/优化日程fresh至450，full＋public各权重1；
已有T450直接复用，候选一次correct400；两者公共策略各做旧36-task/144条train诊断，并读固定A28。
公共增强且完整改善支持有限联合收益；仅公共改善而完整不改善将降低该修复优先级，不能靠内部风险保护它。
本批尚未执行；16GPUh/32GiB、预计3–5小时，无自动900或M改版。结果和实际状态以后续progress为准。
对M已进一步明确：原递推有未来K乘积带来的全局/顺序依赖，不能称无上下文；
局部E内容不直接依据其它时刻重新解释，是具体候选限制，尚非已证实际别名/根因。下一内容设计须保留T正例和LocalField等阴性。

## 254. 全视频Value内容与递推混合是不同依赖，尚未证明哪项限制主要性能（2026-09-30）

§35实际启动期间，机制§79将候选具体为：原native H的逐horizon时间attention形成c，
在原`GELU(PK+C H)`中加入target-specific `U c`，再乘原DΔH；A/K/R/O及唯一38-target输出保持。
U=0包含原T函数、静态H仍M=0；新增信用为`Uᵀ[g_E⊙DΔH⊙GELU']`，可经attention回到其它时刻H及公共β。
固定共线key的抽象反例说明，晚期Value可利用早期证据，而原局部Value在该地址极端下不能；
这不是实际碰撞/语义丢失或全模型不可表示的证据，不据此宣称已定位根因。

clock只改覆盖，虽信用变化明显却未提高主要能力；内容扩展改变了不同计算依赖，但也可能失败。
LocalField已经有全视频上下文及同参数真实cotangent消费，仍48→47近source47；
本候选继承T的公共实际A/native而非裸source核，这项差异尚无新实验验证，旧阴性不清零。
函数包含关系、零静态输出、attention或非零梯度均不能替代绝对闭环收益、能力保持及最终视频证据。
本节没有新代码/模型/GPU任务，只有一个未激活的算子候选；当前唯一训练仍是§35原T双目标fresh450。

## 255. Fresh公共目标改善独立能力，却主要重分配已有完整功能（2026-10-01）

design§35从同identity/事件/日程fresh450，只增加同query公共FM，完整114/400对旧T450122：R88/G26/L34，churn60；
对强MT153为R86/G28/L67。八task簇差值区间对T[−28,+12]条，不能将−8单点诊断为确定伤害或最终上限。
公共train144从68到80，R60/G20/L8；原train24为26/96→36/96，support42/48→44/48。
公共实际收益主要在Spatial/Object；Goal/Long小幅下降，held完整task11/26/31分别少5/6/5例，task6/16多3/4例。
task31并非仅丢掉已经完成的第一物体：cream-cheese ever39→31而final28→29，butter ever21→12/final21→11。
没有held公共读取，不能将公共训练改善外推为held基础变强，也不能拼接旧T1800 full144计算候选视频增量。

main独立重算全部原行、两个完整臂的goal trace及24份A28 FM速度预测，见机制§80与
`docs/analyses/operator_joint_public_evidence_20261001.json`；未重复工程QA或新增forward。
前5公共风险.157547→.139920，完整.138351→.135889；完整50公共.133573→.121166，完整.116173→.116429。
精确函数分解b=公共、c=完整−公共给出`df=db+dc`：前5池化cos(db,dc)=−.81597，
E db²=.0101134、E dc²=.00898568、2E db·dc=−.0155570，完整输出改变能量仅.00354205。
条件作用能量前5 .016786→.005256，两个teacher共同部分.016502→.005006，teacher偏差仅.00028358→.00024977。
这支持“功能主要重新分配”，不证明梯度冲突、知识因果比例、teacher语义噪声消除或唯一参数通路。
本固定train面板和新held结果共同降低“持续公共目标自然解决主要完整能力缺口”的支持；
仍保留合理功能目标、真实公共正例和450早期/共享图未充分优化的可能，不用负分清零完整架构。

当前不追加joint900或lambda/LR/rank扫描。下一active§36将§79候选收紧为单层256维/4头时间attention，
每target零初始化U只进入原Value gate，保持native/实际A/覆盖/38target/双目标；详见机制§81。
新增路径让晚期可用写入依据早期证据决定内容，区别于仅改变Q乘积的clock；T公共执行坐标也不同于LocalField裸source核。
这些结构差异允许有界检验，不证明局部语义歧义已是实际根因，完整上下文的旧阴性继续降低盲目成功预期。
只训fresh450并读correct400/公共train144/A28；比较同时包含joint114、T122及MT153，不只胜新失败对照就称成功。
如果内容路径仅改变风险/内部量而完整能力差额不改善，降低主假设优先级，不自动增层/增宽/续训。

§35全部688 episode/24读出完成并退出，5dd训练与读取；普通10.90634GPUh＋holder保守上界.40036，
记账上界11.30670/16GPUh；留存16.27780GiB、峰值保守界16.77662/32GiB。c202共享prefix仅供后继，来源不混写。
§36合同准备完成、尚未宣称启动；16GPUh/32GiB、整批预计3–5小时，后续真实状态见progress。

## 256. Context450有有限闭环收益，条件作用变化未形成稳定多阶段能力（2026-10-01）

§36完整126/400，对Joint450114为R96/G30/L18/churn48，对T450122为R92/G34/L30/churn64，
对MT153为R94/G32/L59/churn91。八task簇95%差值区间分别[1,23]、[−16,24]、[−65,7]条；单seed区间不包含训练随机性。
主收益在Object42→50（Joint参照），task11回到原T的42、task16达到8（原T1/MT10）。
task31仍9/50但只保留Joint的3例，得6失6；cream-cheese ever/final31/29→35/31，butter12/11→9/9。
task39虽2例曾关微波炉，杯子进炉仍0、完整成功0。没有证实全局内容已经教会多阶段操作组合。
公共train144两者80但R73/G7/L7，原train24同为36/96；不能据公共同分声明公共函数完全相同或held增益已归于视频。

main独立读取所有成功原行、400候选goal trace、36份FM预测及配对1,800个已存训练条件，
并核实际Value算子调用，完整数学解释和证据见机制§82及`operator_context_value_evidence_20261001.json`。
前5公共FM .13992009→.14008095、完整.13588949→.13583746；全50完整.11642899→.11678992。
前5完整motion6变差、gripper改善，224个condition-query仅107个改善；平均误差不能解释所有held增益。
Context−Joint的E db²=.00007322、E dc²=.00039925、E df²=.00036130、cos(db,dc)=−.32509，
主要改变在条件作用而非再次大幅公共/条件重分配，但这不是U/context的唯一因果归因或更大M更好的证明。
新信用从其它时刻H经attention→U/GELU→DΔH→O/K/覆盖→M A→自身执行，实际图与合同一致；
LocalField/clock阴性保留，不能由attention非零或涨分直接宣称操作语义被识别。

新U/attention在第2/3步已有非零梯度；五个90步段的同事件full FM小优势没有在后段扩大。
因此不以零初始化延迟或loss继续下降为续训依据。实际继续理由是：当前相对MT的−27主要由task3/31合计−28构成，
而原T450→900恰在这两项24→43、14→21；一轮teacher覆盖仍不足以裁决它们的学习上限。
原T900原始400行148、T810144已直接重读，机制§80误写的147已修正（147为T1080），不改历史结果。

active§37是一次固定图/目标/Adam的450→900学习检验，900唯一主点；900>MT153时才补既存810的correct400保持读取。
没有Joint900，900相对T的比较包含公共目标与context两项，不能重新冒称独立识别context效应。
若主要强参照差额/能力保持仍未改善，不自动1350/2340、增层/增宽或loss小扫；正效应亦不等于最终方法资格。
§36记账10.17446808/16GPUh，全部544新episode/12新A28完成退出；新存储保守峰值界14.43244/32GiB。
§37合同准备完成，16GPUh/32GiB、预计3–4小时，真实派发/启动以后续progress回执为准。

## 257. Context900获取接近MT总量，但同龄优势与Long保持未扩大（2026-10-01）

§37完整900=151/400，450=126、原T900=148、MT=153；对三者分别R100/G51/L26、R122/G29/L26、R116/G35/L37。
八task簇95%净成功区间[1,59]/[−17,21]/[−31,26]，只描述任务重采样，不含训练seed；未超过153所以810未读取。
suite48/54/35/14、breadth7/8；相对MT Object+8抵消Long−10。450→900候选+25、原T+26，同龄净优势+4→+3。
task3 28→45且只失1；task31 9→14却只保留3、得11失6；task16 8→9也只保留2。不能把总分成熟写成普遍保持。

main直接重算完整/公共原行、候选400 goal、参照task3/31及48份已存FM预测；进一步读取task31四模型body/eef轨迹。
证据`operator_context900_evidence_20261001.json`，完整解释见机制§83；无新增模型forward、梯度或环境调用。
task31 cheese37/33、butter17/17；22例cheese曾达成但butter从未达成，4例cheese曾达成后最终丢失，不是主要在最后保持失败。
旧450丢失6例均仍曾达成cheese，只有1例butter曾达成。cheese-only失败混合未接近、接近未搬运、搬运未完成，
不能统一归因于阶段遗忘/对象选择，MT自身也有21例该类失败。状态/结果条件化与连续几何描述不作独有根因证明。

公共train80→85，R73/G12/L7；原train24 36→41/96，support44/48保持，MT为46/96、47/48。
A28前5公共.14008095→.13368584、完整.13583746→.12812152；全50公共.12138589→.11761492、完整.11678992→.11051816。
前5公共/条件作用变化cos=.09820、全50=.03000，无旧fresh联合目标那种大幅相互抵消；风险分解不是知识或闭环因果比例。
224条件-query中149个前5改善，但task32两teacher完整前5均值.11900045→.12154942，公共却.12452699→.12098221，
条件作用风险收益从+.00552655转−.00056721；同task完整全50仍改善。实际私有Writer信用为.1∇R5+.9∇R后45，
这给出一个真实条件分支上的功能取舍，却不证明逐步梯度冲突或整个held缺口的主因；旧端点/前缀/learner-state阴性与正例都保留。

降低“完整上下文Value自然形成明显更强迁移/保持”的支持；保留实际+25获取，不清零整套架构。
本轮不追加1350/2340、加层/改A/回读/辅助loss。原T1800 full105/144不能代替新Context900，当前85只测公共、A28只测专家query。
active§38仅补固定900的完整train144，与既有公共85和MT93同场景比较，改变获取不足/关系迁移受限的解释优先级。
不把该诊断设为新资格门槛，不因任一结果自动扩矩阵或启动新方法；仅144新episode、0学习/0FM，2GPUh/8GiB、预计20–40分钟。

§37训练/读取17ee3e38，父5f4f76e7，原状态完整恢复，新增50400 full及同query公共，36task第二轮各50teacher。
544新episode/12新A28、36旧A28复用，9.97469956/16GPUh、首次完整验收约2.928小时，GPU全退出释放。
最终completion观察高水14.02899GiB、保守界14.52284/32GiB；CPU来源字段修正无GPU重跑/新增成本，原失败保留。

## 258. Context900完整已见能力103/144，条件增量明确而held功能分工仍未知（2026-10-01）

§38完成，main重算四模型144原行及576份既存goal trace，无新forward/梯度/环境；证据
`docs/analyses/operator_context900_seen_evidence_20261001.json`，完整推导见机制§84。
完整103、自身公共85、MT93；train24为59/41/46，support12为44/44/47。旧T1800105仅不同图/目标/年龄背景。
对公共R82/G21/L3、churn24/J.773585，对MT R85/G18/L8、churn26/J.765766；whole-task95%净成功区间[9,27]/[0,21]，非seed区间。
train24对公共13task净正、0净负，净增18全在原24；仍有task19/state33、task20/state34两行损害，support另失55/state32。
完整breadth35/36，原24仍失败37/96，四状态/task不能证明充分掌握或正式400资格。

Long完整10/24对公共5/24、MT8/24；32/34/35/36/37/38分别3/2/2/1/1/1。
task32完整/公共都四次开炉，完整放壶3、公共1；36/37/38各新增一次兼容目标完成，当前M可在自身后续状态补出操作。
34仍比MT少一，38同为1却成功state不同；不能称统一的阶段失败已经解决或显式组合知识已经获得。
先前task32 A28前5完整略差于公共，未预测这里3对1的闭环排序；不据其局部风险反例改loss。

本结果降低整体获取失败的解释，但不同难度train/held不能直接估计泛化差额。匹配任务上有精确成功数分解
`full−MT=(full−public)+(public−MT)`：原24为+13=+18−5，support为−3=0−3；held总−2及Long−10的两项仍未知。
若当前公共较强而M损害，方法重点应是条件迁移；若公共较弱、M有益，则不能把完整不足概括成M破坏或没有迁移。
旧T公共、当前train公共均不能填当前held公共；恒等分解也不是知识或中介占比。

active§39只补固定900公共validation400，复用完整151/MT153，0更新/新FM/A28，不改选checkpoint或自动训练。
当前未发现足以重开回读、更多任务、前缀辅助、加宽A/O的具体新依据；完整历史正负例保持。
§38训练17ee3e38、读取e4c33253；.53366170GPUh，实观测5.51342GiB、保守界6.21341GiB，所有消费者退出释放。

## 259. Context900公共134与完整151，Long缺口不能只归于条件破坏（2026-10-01）

§39完整结束，main复算三模型400原行及1,200份goal trace；证据`operator_context900_public_evidence_20261001.json`、机制§85。
公共/完整/MT总134/151/153，suite分别36/52/33/13、48/54/35/14、48/46/35/24；breadth6/7/6。
完整对公共R112/G39/L22、churn61/J.647399，公共对MT R106/G28/L47，完整对MT R116/G35/L37。
精确whole-task重采样95%净成功区间[−5,44]/[−46,6]/[−31,26]，非训练seed不确定性；
执行者10,000次重采样公共对MT上界5属抽样差别，未改变任何成功行。

总差额−2=+17−19、Long−10=+1−11；这是成功数恒等式，不是知识/中介比例。
公共自身在主要Long差额中已弱，不能把完整不足统一解释为M破坏了强公共；净条件收益又主要在task3+12、11+6，task16−4。
task31公共13、完整14、MT24，完整对公共R4/G10/L9；新增10条公共均曾完成奶酪、从未完成黄油，完整补齐两项。
失去9条中完整只1条曾完成黄油、7条曾完成奶酪；正反例排除“所有条件迁移均无用”和“只差保持第一目标”的概括。
奶酪/黄油ever为公共45/16、完整37/17、MT45/25，final33/15、33/17、42/24；完整两项均从未达成11条，对照均4条。
单物体11/16公共39/13高于MT36/10，双物体31却13对24；布景/语言/horizon不同，不能据此确诊纯逻辑组合失败。

公共目标、后置全视频Value及继续同池学习没有扩大成熟优势，降低这几项单独充分修复的支持。
β-native→X/H/context→动态Value/覆盖→M A→自身执行的实际链路已有获取，不是死图；保留真实后续操作增量和不利任务。
公共134低于MT不等于梯度冲突：100,800不同逻辑query与MT172,800的曝光/优化轨迹不同，同query公共重算不是翻倍数据。
本次未换checkpoint、未触发810/新controls/Test/RL；下一架构取舍须独立解释实际新依赖与可失败预测。

训练17ee3e38、读取96d6ce67；原76公共因子直接复用，0更新/新FM/teacher native；成本1.25055551/2GPUh。
首attempt276有效行后首动作前RGB一像素一级差，原因未确证，原校验不变；既有队列恢复剩124一次，失败和未发布捕获保留计费。
最终400全配对通过、消费者退出释放；实观测1.01262GiB、阶段保守峰值1.8/4GiB，非连续实测峰值。

## 260. 共享参数两次native读取是有历史反证的联合学习假设（2026-10-01）

机制§86和active design§40冻结唯一干预：`M0=Fθ(Nβ(z))`、`M1=Fθ(Nβ⊕M0(z))`，最终β⊕M1。
F为当前Context完整算子，两次同一参数、memory各从零，原公共A/B0不被临时覆盖；没有M0+M1或两个部署adapter。
全视频先经M0改变每帧真实action_in/Q/V计算，再改变X/H、K、动态Value和覆盖；不是仅在已形成H后面增加上下文。
局部`ΔH1=ΔH0+(J_N,t+1−J_N,t)[M0]+O(||M0||²)`说明新增算子响应项，不能赋予未验证的对象/操作语义。
action_out输出仍被native丢弃，第二次仍state-free、固定probe/τ1；M1也未要求等于M0，不能称读写完全一致或收敛。

真正信用为`DθM1=∂θΦθ(M0)+∂MΦθ(M0)DθΦθ(0)`，最终FM经第二次native回到M0与第一次F；
公共执行A/B0的直接/间接贡献和公共FM保持，不detach中间条件，不加中间loss/标签或task-local优化。
O=0初始函数/一阶信用代数上恢复单遍，新回读局部从O(ε²)起；第一步中间信用弱不等于bug，也不保护长期低分。
参数数目、初始化/事件/目标不变，唯一反事实是多一次带条件的native计算并联合学习。

§27/65冻结T替换式回读前5风险均值+.00521058、8条件仅1改善，§71缓存末层拟合和直接B跨episode/留任务均失败；
Unified和LocalField的原生条件/全局内容阴性也保留。因此不直接改装Context900、不做回读缩放、多迭代或缓存拟合小扫。
在公共目标和后置context未扩大成熟优势、实际条件后续操作已有正例的证据下，main选择一次fresh完整图检验；
这不是已确诊主要根因或因fresh未试而清零历史成本，旧联合回读想法在§20/65已经提出。

唯一fresh450、原36task及50,400 full query＋同query公共项；主correct400同时比较Context450126、原T450122和MT153。
公共train144及20份train A28解释分工；8条teacher的两次H/三处AX/M只被动保存，无新增层扫描/held标签/环境诊断。
只改变内部量或FM而未改善完整能力/主要缺口时结束本批，不自动900/第三遍/blend/另一Writer；正例另看最小邻点保持。
合同24GPUh/36GiB、预计含实现4–6小时，profile后校准一次；实际派发/训练状态以progress和运行原件为准。

## 261. 两遍联合学习改变了动态描述，但没有改善完整能力或主要缺口（2026-10-01）

§40完整结束。fresh450的β⊕M1为116/400，Context450126、T450122、MT153；suite25/43/37/11、breadth7/8。
对Context R96/G20/L30、churn50/J.657534；对T R95/G21/L27、churn48/J.664336；对MT R92/G24/L61、churn85/J.519774。
主讨论精确whole-task95%差额为[-24,3]/[-18,7]/[-77,-3]，非训练seed区间；不能称普遍显著劣于Context。
证据`docs/analyses/operator_self_read_evidence_20261001.json`，完整特征—算子—功能—历史解释见机制§87。
main复算2,176条原始成功/goal记录、300份既有continuous、8份native；A28复算44预测/4target，main另核16原预测。
没有新模型forward、GPU、梯度或环境。

对MT主要差额仍为task3−18、task31−13，合−31；Context对应合−28，不能以task31对Context+2称主要缺口改善。
task31对Context R4/G7/L5：新增7行Context均完成过奶酪但从未完成黄油，是真正补出目标的局部正例；
失去5行Self-read却无一曾完成黄油，不能只解释为完成后没保持。奶酪/黄油final从Context31/9变为26/13。
task3对MT失去21行中，17行目标碗位移≤2cm，其中11行另一碗移动>5cm；Context自己的MT失去17行也有14/10。
该行为与目标操作绑定不足相容，但位移含碰撞，不能唯一定位视觉/语言/Writer/执行机制，更不能只归为长任务顺序。

公共train14483，对Context80 R74/G9/L6、对T68 R60/G23/L8、对MT93 R77/G6/L16；train24为38/96、support45/48。
本批没有完整train144与held公共，不能从小幅公共变化推导完整获取充分或held公共/条件贡献。
两次native的Hbar相对变化2.07%–9.10%，实际ΔHbar变化17.18%–50.67%，三处M变化约8.5%–50.2%；
Q8/V8单位地址均很接近，action_out变化更大，但这不是key/Value因果隔离或语义证明。

A28最终M1对公共β全50风险改善3.85%、前5改善2.92%，条件分支确有功能作用。
额外M0→M1全50只改善.000030525（0.0262%），前5恶化.000216825（0.1596%）；teacher平均query仅50/112、55/112改善。
精确分解全50为−.000308741+.000278217，前5为−.000108033+.000324857；新增输出变化没有形成强而普遍的纠正。
task0两teacher均恶化，task20全50改善却前5均恶化；task32前5平均收益约98%来自一条query。
M0无独立训练/loss，不能拿它当另训单遍模型或选择部署；first5仍是单步FM切片，不能冒称闭环动作风险。

本批把§65/71保留的fresh联合图真正检验为有限阴性，进一步削弱当前“多一次条件native足以修复”的预测；
不再仅以旧冻结图没共同训练保护这一假设，也不否定所有回读、当前条件作用正例或新模型的数学可达性。
shared A/深Transformer仍是独立问题：可逆换基不改变行空间，teacher/执行同A不要求跨task同A；
但M=0包含同rank公共LoRA，不能把共享A直接当低于MT的不可达理由。本批未验证128不足或深Transformer更好。

训练/读取da758e6b，544新episode/20预测/8native完成退出，90..450完整ECP保留；总17.092523GPUh，训练15.022237。
存储观察14.766659GiB、保守32/36GiB；事件接续与CPU字段解析恢复保留原件，未重启训练/改科学计算/重跑GPU。
Owner最新要求分析后先讨论：§40结束、无active后继design，不启动续训、第三遍、补评或架构改动。

## 262. 公共任务表达与动态内容写入的完整综合（2026-10-01）

Owner重新授权深入分析及必要分析实验，要求完整解释和一个最合理方案；长期巩固/merge是动机，不在本轮实现。
参数量没有上限，不以减少参数为由偏好循环。完整推导与最近似历史见机制§88，数字与来源见
`docs/analyses/t_public_basis_synthesis_evidence_20261001.json`。本轮0新模型前向/更新/环境/GPU。

A0/B0可理解为已见任务的公共参数表达，公共policy仍按语言与自身观测调用不同规则；
M是共享Writer一次前向产生的快速任务适配。full-only不独立确定公共/条件功能分工，公共目标是功能要求，非涨分保证。
把已经学得的B0A0合并进source本身保持函数，不产生额外能力；真正的新能力须先经经验巩固，保持旧任务亦非直接平均M可保证。

每帧有50个H，整段为N×50×1024；当前fixed probe/tau1得到公共动作计算响应，未保存速度或执行十步采样。
同j的跨帧Delta H是公共响应变化，不是教师正确动作减公共预测，也不是已证时间错位bug。
8份self_read450原件中，Delta H投到所有50槽相同子空间的能量第一遍22.13%–56.93%、第二遍17.09%–57.61%；
不能说50槽已经等同或应平均，也不把其它能量当有益语义。

main直接读取T2340四train task×两teacher的实际LoRA余切和父A。
执行A梯度在row(A)外的能量占A28全50的86.17%–89.78%；128/1024维等方差参照本为87.5%，不称信息损失或容量根因。
该外部负梯度向互斥episode B20十步生成前5风险迁移时，只3/8呈有益一阶方向；A28前5只1/8。
行空间内方向也未普遍迁移。这仅降低“放开A便能修复”的简单理由，不否定条件A共同学习，更不构成Transformer阳性。

更具体的结构限制是：T和Context均将每个Value乘本地同horizon的D Delta H。
知识需要来自视频动态，不等于每个应调用修正的状态都必须有本地非零变化；局部门控把这两件事绑在一起。
已有全局gate、change-clock、两次native均没有直接解除这项内容来源限制。
一个固定特征/地址CPU见证以满秩A=I、写入地址e2/e2/e1和最后局部零变化，得到原/clock在e1作用均0；
允许将早先动态作为最后Value则可写出.75。不称真实LIBERO已满足该配置，也不称整网无法通过其它学习回避。

主讨论首选完整候选：保留公共native、实际A读写及原delta-rule，以一次深Transformer联合处理全部时间×horizon。
c保留公共响应上下文，d以真实Delta H初始化，通过c选择的attention和对零保持FFN传播动态内容；
最终Value使用全局d_final，不能再乘回局部Delta H。首版建议4个独立1024宽block，数值是实现起点而非已证最优。
最后仍(A0,B0+M)一套38-target rank128，fresh共同训练，沿已有full＋public双FM；无teacher标签输入、无部署优化或第二adapter。
这能把证据出现处与修正应调用处分别学习，并使整段无变化时M为0；后者仍不保证有益动态语义。

旧Horizon/Unified、LocalField、P/I、Context的完整阴性仍约束该选择；新方案不是因为更深/参数更多便获得资格。
尤其Context900151与原T900148未形成成熟优势，降低仅增强context便足够的确定性。
本轮未确诊唯一性能主因、未实现或训练候选、无新的formal active design；交付是受现成正反证据约束的明确方案。

## 263. A0可通过视频适配，但须连同后续M的实际坐标共同学习（2026-10-01）

Owner要求继续仔细推导条件A。完整推导见机制§89，同一综合证据文件追加task_adaptive_A_followup。
本轮只有源码/历史核查与合成小矩阵CPU计算，没有新PI05前向、数据读取、科学模型更新、环境或GPU。

公共A0/B0仍跨任务共同学习，视频可生成S/M，最终`(B0+M)(A0+S)=B0A0+MA0+B0S+MS`。
因此M=0时S也可改变公共输出侧的触发；B0(A0+S)不应继续叫无视频公共函数。
左乘R不扩大执行输入行空间，但可改变完整Writer的归一化地址度量，不能称完全无效。
S(I-P_rowA0)非零才代表新输入方向；这只是分析条件，不施加只许正交变化的硬约束。

完整候选保留公共beta一次native、完整时间×horizon动态解释；输入侧头结合真实高维X、c及全局d产生128维内容，
用真实X归一化地址关联写成128×d_in的S，然后以A=A0+S重新形成B侧key/Value/覆盖，最终部署同一A和B0+M。
B侧Value额外读取已生成的S X，能知道A造成的实际读取变化；它不是teacher标签。
两侧Value都使用全局d，不再乘回本地Delta H；全动态为0时S=M=0，退回公共函数。
保留一套38-target LoRA，首版rank128是承接消费者的选择，不是参数上限。

输入S的行空间仍由该视频的真实X支撑；未保证这些方向具有操作语义或匹配自身hidden。
其X/H来自beta，而不是适配A后的第二次native，同A不等于两端特征分布已对齐。
真实A信用同时有执行直接项B^T G_W及经M的间接项J_(M,A)^*[G_W A^T]，完整图不断开；public/full共同FM保持。
合成图验证两项和与autograd差8.24e−18、遗漏间接项差范数.009623；仅为代数，不说明真实量级或效果。
双零出口的合法identity可在B更新后取得A侧信用，无须阶段预训练；不叠加额外零门导致永久断路。

最强旧反例不能清零：LocalField早已有实际X关联生成A、完整上下文及同Ur功能监督；
Pullback已有每视频PCA A同时用于B写入和执行；P/I已有fresh公共128＋条件A/B16单adapter，后期未保持。
本次实际区别是可学公共native、保留A0的动态S以及该A再次参与M生成与自身执行的完整共同学习。
它并非首次条件A/真实X/乘积交叉项，也未获性能阳性。旧固定B的3/8局部方向诊断不能替代这张新图的学习比较。
候选尚未实施或训练，无新formal active design；未证实A是主瓶颈，也未证实该候选优于§88固定A方案。

## 264. 因果表示与片尾编译分开；公共目标可用随机单分支估计（2026-10-01）

Owner追问时间因果、公共loss额外成本与整套架构。机制§90给出讨论建议，未启动实现或训练。
严格因果须同时使用帧块下三角mask、后向差分H[t]-H[t-1]及不回灌未来的归一化/上下文；
同帧50个horizon槽可以互看。只加mask而保留前向差分仍会泄漏未来。
转移在到达帧t才可用；若按T的起点寻址，配对缓存X[t-1]，不得混同终点X[t]，写入索引须显式登记。
最终A(V)依赖整段视频，再用于M的重放；所以是因果编码、片尾编译，整个Writer并非逐帧在线因果。

同teacher/task内随机均衡分配full/public query，每条只预测一支，按逆概率加权可无偏估计原L_full+L_public。
公共支必须同时关S/M；仅关M仍含B0S。full信用回所有相关模块，public直接教公共beta。
固定query数时各目标有效样本约减半，梯度估计方差与优化轨迹改变，不能承诺同样性能或总训练成本。
当前dual_functional_credit已经共享冻结prefix，节省主要发生在重复action-expert suffix，不是完整模型的2倍加速。
只保留完整loss、正则或detach不能普遍保证无视频公共函数有效；单分支方案保留真实公共功能监督。

## 265. 公共能力应由功能学习形成，不能把条件因子吸收当作知识巩固（2026-10-01）

Owner追问S的几何依赖、吸收S/M后混合训练及更清楚的公共学习机制；完整分析见机制§91。
S不必先求A0行空间，但有用调整依赖当前A0作用；§89已有间接beta-native联系，显式A0 X可使相对编辑接口更清楚。
把目标写成A0 Xi+Delta U后，响应纠正形式严格回到原S delta-rule；不需要硬正交投影，也未证明新输入带来性能收益。

完整接收S/M会把一个条件策略变成新公共函数；后续混合训练才负责跨任务功能筛选，并非赋值自动完成巩固。
部分吸收的因子插值相对作用插值相差alpha(alpha-1)MS，E[S]=E[M]=0也不消除E[MS]。
Reptile近似要求真实任务优化等条件，生成S/M不能直接当作公共风险的梯度；适配初始化与无视频公共最优也不同。
beta改变还影响native与Writer后续输入，不是给同一固定生成器免费重定基准。

实际native.py已经调用公共action_out_proj并丢弃结果。复用其固定probe/tau1预测、以授权teacher episode动作作训练标签，
可给公共beta直接辅助信用而不为此新增完整policy前向；动作不能进入Writer条件，完整跨episodeFM保留。
当前teacher无state、执行有state；固定端点/噪声仅为实现选择，不是方法边界，Owner已明确纠正。
可以从随机noise沿公共策略逐步去噪，合法读取各tau的hidden，此时缺少state是主要输入差异；
但需多个suffix前向，并明确生成链的监督/梯度，不能沿用“一次native现成输出等价公共FM”的说法。
标准FM的x_tau=tau*noise+(1-tau)*a在tau<1时含真实动作，其hidden不能进入Writer；训练专用FM分支可用，须与读取条件分开。
因此该辅助尚不等于原公共执行FM，也未证明闭环能力；推理式读取与标准FM训练应分别论证，详见机制§91.3补充。
既有公共FM、同视频完整LoRA辅助、冻结source纠正及功能蒸馏是不同近邻；未在定向范围找到该公共native消费者的正式结果。
当前不采纳直接因子吸收；native辅助只列有依据的低成本候选，公共执行功能仍须真实证据，未定新formal配方或启动实验。

## 266. 公共功能分工可由条件作用的软收缩推动，不必另算公共FM（2026-10-01）

Owner指出T的完整训练已能改善公共基础，所需是尽可能避免S/M重复承担公共部分。此前过度强调独立公共能力的普遍保证，
没有充分讨论可实现的优化偏好；独立public FM并非唯一机制。完整论证见机制§92。
原T只有full FM，当前Joint只有full/public两项；统一AdamW作用于生成器等参数，不等于对每视频生成的条件作用收费。
候选为L_full+lambda*E||delta_W_l sg(h_l)||²，其中delta_W=(B0+M)(A0+S)-B0A0，并保留实际LoRA缩放。
展开为B0 S h+M(A0+S)h，在已有完整query的执行hidden上计算；不需要第二次公共policy前向，也不引入teacher标签条件。
主FM仍完整反传；该正则仅把采到的h当本步固定参考，约束本层作用，不能冒称完整策略动作差。
公共/条件可重分配而完整函数相同时，该代价偏好公共承接共同作用；线性多任务平方损失例子给出b=mean(y)、m_t=(y_t-b)/(1+lambda)。
它同时收缩必要任务差异；低rank、共同生成器、native依赖与非线性均限制将线性结论直接外推。
不以S/M raw norm、均值或跨层低范数证明知识分工。均值残差惩罚可另作候选，但有限task batch混入方差，且零均值不等于无公共行为。
定向核查未发现此前实际训练过该项；旧视频内中心化、零code锚定及冻结SHRINK诊断均非同一干预，不清零其各自边界。
小矩阵仅核展开、换基不变性与线性最优条件；候选没有真实模型或性能证据，不因此启动训练或删除现行公共目标。
