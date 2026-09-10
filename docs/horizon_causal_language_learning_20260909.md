# 后端条件路径：fresh学习对照

**最新授权上限（覆盖下文此前探索授权与条件追加计划）：** 2026-09-10 CST Owner最新限制：当前已启动的 `horizon_causal_within_target_rank_shared_seed7_20260910` 是本次分析最后获准的完整训练，仅完成既定400步及200/400评测。此后分析排查不得再启动完整训练或同等规模的重训练，也不得通过改称探索、拆分短段、追加seed/候选或延长当前run绕过限制。后续使用已有checkpoint、历史证据、冻结评测及有明确问题和小预算的轻量诊断；证据不足时明确报告未识别项，不以必须查清为由追加重训练。正式方法修改/采纳及未来正式训练仍须等待Owner复核后另行明确授权。

2026-09-09，结果前登记。归属`horizon_causal_learning_plan_20260909.md`，本文各面板仍为原因分析/探索证据。2026-09-10 CST Owner最新明确恢复分析后的复核停止点：完成原因分析及必要隔离探索后，停在正式方法修改/采纳和下一次正式训练之前，等待Owner复核后重新授权；此要求覆盖此前睡前的扩大授权，不改变下列既定面板、判据或证据等级。

## 综合解释与证据边界（共享训练结果待补）

截至Compiler固定确认完成，最具体的原因证据是：**当前Compiler首次额外仿射查询支路改变了学习所得策略，关闭它能改善本协议下的验证总分；但它只解释部分Object/Long收益和部分BBQ保持，不是广泛迁移与稳定性的完整修复。** 原all与local_h_read的400严格同初始化差为init7的90→126、init11的108→119；固定init7换正确视频关联为82→126。两个候选自身200→400均净增长，尚无平台证据；init11同时丢37个成功、BBQ丢20/35，故不能把净增说成稳定保持，也不能用126接近某个历史分数来宣布旧新差距已经解释。

完整因果链是：exact language与有序真实视频先进入冻结vision/Gemma，Action Meta从合法原生prefix取得完整50H响应；逐帧条件引导四组局部对应、两端视觉读取及过去时间组织；末组P4提供Compiler的Value；Compiler把它映射到target/rank条件，再由native D生成唯一38-target完整LoRA；冻结source在机器人自己的状态上执行。前端、Compiler和D的职责分别是取得可迁移证据、把证据编译成目标因子、实现有效策略修正。图上有梯度或存在信息通路，只能证明可学习连接，不能证明各层已经兑现职责。[当前数据流](concept.md)

| 问题所在 | 当前有区分力的证据 | 可以支持的结论与未识别部分 |
|---|---|---|
| Compiler额外查询入口 | 固定local的2×2 fresh学习中，关闭Compiler在H-read开/关背景的200效应为+5/+16，400为+36/+20；独立初始化与换视频关联保留净增方向 | 有局部学习因果贡献。干预同时关闭query_language权重与bias，不能单独归因文本内容；H-read删除没有相同的一致方向 |
| 未见任务能力保持 | 多版BBQ在200后丢失，而训练任务held-video行为继续增长；init11 all其它七task73→103抵消BBQ27→5 | 需要区分获取与保持。不能概括成所有任务同步遗忘，或把总分上升当作原成功未丢失 |
| 条件与目标选择 | 固定四state的all400回放操作绿色干扰瓶；init7候选400围绕正确BBQ操作，三例完成、一例未完成运输 | 支持这些实例的目标选择改变；不是全局失败率，也不唯一定位前端、查询或LoRA内部机制。一次候选回放与历史成功标签分叉已保留，未重复追成功 |
| P4→Compiler→D的功能获取 | 既有train task7局部free-C与A/B为4/4，normal与P4-only为1/4；全八task C仅20/32对normal18/32 | 当前冻结D能表达部分normal未取得的有效修正。不能由局部可达性排除上游信息获取、Compiler读取或优化困难；不能把局部oracle当容量上界 |
| 输出独立性与共享 | 当前同target跨rank D绑定训练正在运行，复用原all的科学协议；历史Target-Owned已有99/76/86/68负例 | 200完整train96为32对41、validation115对103；实际生成B/BA近单方向，400待完成。即使后续有收益，也只识别参数绑定整体，不能拆称纯参数量、纯正则化或已证明跨初始化稳健 |
| 拟合与实际执行 | 完整native FM与10步示范动作MSE仍不能可靠排序闭环；固定Long36/38已转向第二对象仍未完成，既有专家同状态也失败 | 平均示范拟合不足以裁决策略可用性；这些实例不支持全部归为目标选择错误。状态覆盖、接触与恢复机制尚未单独识别，不能直接宣布换loss或RL有效 |

前四项主要原件见本文后续固定学习/确认矩阵与回放；局部接口和Long执行证据见[完整冻结与局部诊断](horizon_k1_causal_diagnostics_20260909.md)。冻结依赖、局部oracle和fresh学习解决的问题不同，结论不能相互替代。

**历史分差须按实际缺口解释。** v5.2 old900的132含Spatial19，当前init7候选126仅Spatial1；总分接近仍缺旧模型取得的关系任务能力。v6 old500为121、task-complete400为143；同类纯FM、4-task更新曾取得较强结果，反对把FM或四任务组织判为普遍无效。与此同时，旧新架构、Meta范围、teacher/query池与抽样、LR/更新分组均不同，旧模型未显式排除teacher/query同episode。历史不是当前严格配平的反事实，不能把Compiler的+36换算成解释历史差距的百分比。Goal23与Long32在旧曲线也长期弱，属于共同未解决能力，不能全算作Horizon新丢失。[历史协议审计](horizon_k1_evidence_review_20260909.md)

**交付尚缺最后共享run的200/400证据。** 按最新Owner上限，仅完成这同一轮既定checkpoint、validation400、held-video train96和held3072，并报告同节点与自身相邻的逐task/suite、breadth、保留/新增/丢失及churn。结果用于确定当前输出绑定是否有帮助、帮助何处及代价；不追加初始化、组合候选或延长训练。随后合并本节原因解释、实测候选和未识别项，停止正式采纳与下一轮正式训练，等待Owner复核。若最后一轮仍不能分离旧新训练配方和全部前端机制，明确保留该不确定性，不以诊断预算已用完宣称唯一根因已确定。

## 假设与可区分结果

当前contextual200/400是同结构下真实训练获取与保持参照。冻结三处置零40/96不能回答从头不给这些条件后能否学得更好。本实验测后端额外条件的学习效应，不测去掉exact language，不等同视频动态必要性controls。

| 臂 | local关系query条件 | H-read query | Compiler首次检索 |
|---|---|---|---|
| all（已有基线） | 当前帧contextual task-token代码 | 同代码经read_language | 纯文本代码经query_language |
| local_only（新fresh） | 保留相同代码 | 线性层输入零，bias继续学习 | 整个额外language route零，包括bias |
| none（新fresh） | 条件字段零 | 线性层输入零，bias继续学习 | 整个额外language route零，包括bias |

保留原生exact language、真实双模态prefix、全部visual tokens、native50H、四组关系/长程/回写和唯一完整LoRA。contextual代码含视觉条件信息，所以none是移除后端额外条件的联合干预，不能被称作纯文本language-only测试。local_only对none只改变local条件；all对local_only联合改变两个检索条件，不据此区分H-read与Compiler单独因果。

共同参数均按原顺序初始化，保留已关闭分支的参数声明以维持共同初始化与state_dict结构；关闭入口后它们不再提供condition-dependent作用。H-read bias是合法共享可学习query，不设随机新query。Optimizer仍按既有参数顺序工作；unused零梯度/weight-decay不提供信息通路。

- none相对all在固定相邻节点稳定提高且保持广度：支持后端联合条件在当前学习协议下有害；需local_only/后续分离臂辨认是local内容还是检索路径。
- local_only好于none且all：支持local上下文有用，同时联合检索条件在当前协议下妨碍学习；仍不能给两个检索分支分别定责。
- none与local_only均更差：反对这两个具体联合删除作为修正，不排除单处正负抵消；是否补单处fresh由剩余行为分歧决定。
- train提高而validation无改善/更差：不算迁移修复；FM下降或局部4/4不改变结论。
- 效应小、相邻反向或高churn：未形成可靠方向，先检查是否由具体task/视频/初始化解释，不能挑峰值或声称没有影响。

## 共同学习与评测

基线复用`runs/outputs/horizon_k1_frame_contextual_v1_seed7_20260909`，运行面9abc9b95。新代码仅在`.codex/worktrees/horizon-causal-learning`，分支`codex/horizon-causal-learning`。baseline all路径除新增显式配置字段外数学计算不变；新臂模型参数结构不变，none/local_only在同一个隔离实现中由有限枚举指定，不加入main正式运行面。

新臂各fresh seed7、Writer+Meta共同学习，source始终冻结；train24、teacher0–15、query16–41、独立query42–45、held-video46–49不变。每update按suite抽4task，每条件64query，SUM后一次AdamW/clip。LR3e-5、warmup8、betas(.9,.95)、eps1e-8、WD1e-4、clip1，与既有基线相同。每条件FM time/noise正常重采样，保留实际task/video/query/RNG日志，事后按语义事件核对三臂曝光。

各臂锁定200/400两个节点，共400updates、102400queries、1600teacher条件；200及400各做完整train24×4state=96和validation8×50state=400。训练侧teacher46–49按既有state映射；validation原canonical无放回teacher映射，官方执行与strict pairing保持。各节点独立3072query诊断只解释动作拟合，不选择checkpoint。自动动作诊断不再重复identity0，因为原生起点相同；不改变学习sampler。

允许根据现场资源用1–4个有用rank、不同physical microbatch；逻辑权重/条件不变，fresh模型初始化seed不变，接受普通BF16/求和次序低位差异。每臂开始后exact-resume锁其topology。记录差异，不以逐元素一致为由重跑baseline或固定batch1。

先做真实短学习profile，确认关闭路径、source冻结、Meta/视觉/完整decoder仍有功能梯度，并检查93帧最长训练视频与真实吞吐/峰值。profile不作能力裁决，不作fresh初始化，不进入正式或探索成绩。已有执行框架复用，新增schema/mode明确exploratory；它不能被当前正式checkpoint inspector误认。

保存200/400完整学习状态、原始行、配对/完成证据。两臂训练+评测预计新增小于40GiB（四checkpoint约16.6GiB、四validation banks与train约9GiB、短profile和余量）；launch前按真实quota和设备登记，而不是按此估计直接启动。

## 当前实现检查与下一步

小模型直接检查已确认：none的group输出对额外条件变化不变；local_only保留local条件作用；两个新臂Compiler对额外文本变化不变；原生response、visual和H-read共享bias仍有梯度。训练循环、数据采样与functional FM不改。当前正式源码/配置保持原状。

两个fresh臂已在冻结探索运行面e60a7ca0完整完成400次更新，200/400学习状态和独立动作诊断均通过检查；实际曝光与基线逐项匹配。profile、launch及资源原件在`runs/analysis/horizon_relation_writer_20260908/causal_learning_20260909/language/`，后续动态状态见progress。

## 首轮语言200节点完整闭环证据

四个新增面板均完整exit0，全部worker返回0；原生strict checker确认任务、初始状态、teacher映射、执行协议及RNG配对。下表R/G/L均相对同节点contextual基线，suite顺序为Spatial/Object/Goal/Long。训练面板每suite24行，validation每suite100行。

| 面板/臂 | 成功 | S/O/G/L | breadth | R/G/L | churn | Jaccard |
|---|---:|---|---:|---|---:|---:|
| validation all | 103/400 | 1/54/37/11 | 7 | — | — | — |
| validation none | 108/400 | 2/58/41/7 | 7 | 81/27/22 | 49 | .6231 |
| validation local_only | 114/400 | 3/62/36/13 | 6 | 77/37/26 | 63 | .5500 |
| train96 all | 41/96 | 11/11/13/6 | 18 | — | — | — |
| train96 none | 39/96 | 9/12/13/5 | 16 | 27/12/14 | 26 | .5094 |
| train96 local_only | 46/96 | 13/14/12/7 | 18 | 32/14/9 | 23 | .5818 |

validation全八task（global ID）结果：

| 臂 | 1 | 3 | 11 | 13 | 23 | 26 | 31 | 32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 1 | 1 | 35 | 23 | 0 | 41 | 5 | 2 |
| local_only | 0 | 3 | 28 | 34 | 0 | 36 | 12 | 1 |

local_only在两个面板均有小幅净增，none没有提高训练任务整体行为；但这是第一个固定节点，尚无相邻保持证据。local_only的validation广度下降且新增37、丢失26，Spatial总分仍仅3/100、Goal task23仍为零，不能以总分+11解释成整体迁移修复。none的train Long从6到5，原6成功全部丢失、另有5新增，也表明总数接近并非保持稳定。以上结果不支持提前恢复全部正式训练、宣布后端条件为主要原因或追加小参数搜索。

各臂逐task、suite、完整配对与完成记录位于`language/{arm}_step200/{validation,train96}_analysis/`；`completed_summary.json`给出汇总，`vs_contextual_same_step.json`保留成功集合和原始比较。剩余裁决仍按预注册完成400的validation400/train96及相邻对比，不按200较高单点选候选。

## 400节点完整证据

两个train96均完整exit0、全部worker返回0；与contextual400及各自200的实际strict配对通过。none为57/96，local_only为56/96，基线为49/96；三者breadth均20。

| 臂 | S/O/G/L | 相对基线R/G/L | churn / J | 相对自身200 R/G/L | churn / J |
|---|---|---|---|---|---|
| none | 14/18/17/8 | 43/14/6 | 20 / .6825 | 35/22/4 | 26 / .5738 |
| local_only | 15/16/18/7 | 41/15/8 | 23 / .6406 | 36/20/10 | 30 / .5455 |

因此200的局部描述不能外推为none没有学习收益：到400，它从39升至57并扩大成功任务覆盖。local_only从46升至56，但200相对none的训练任务优势没有保持。这些结果支持额外条件的影响随学习进程变化，尚不支持local条件有稳定正效应或两处检索条件就是主要原因。独立动作FM均值仍由基线略优，而400训练任务闭环由两条简化臂更好，进一步说明该动作指标不能替代行为裁决；不由此宣布FM目标失效。

400直接none→local_only的train96比较为R/G/L45/11/12、churn23、J=.6618；总分接近仍伴随成功得失。原件`local_only_step400/train96_analysis/vs_none_same_step.json`保留实际配对和成功集合。

收益参照也需限定：上一版first-query-only的train96为200步46、400步59。补充实际strict比较后，400 first-query→none为R/G/L46/11/13，→local_only为44/12/15；因此57/56是相对contextual49的恢复，尚未超过上一版59，不能包装成解决了Horizon总体能力缺口。200对应比较为33/6/13和35/11/11。四个原件位于各`train96_analysis/vs_first_query_same_step.json`；这是已有行为参照，主因果对照仍以当前contextual为共同基线。

两个validation400也已完整exit0，全部六worker返回0，实际同节点与相邻strict配对通过：

| 臂 | 成功 | S/O/G/L | breadth | 相对contextual400 R/G/L | churn / J | 相对自身200 R/G/L | churn / J |
|---|---:|---|---:|---|---|---|---|
| all | 90/400 | 2/44/35/9 | 6 | — | — | — | — |
| none | 92/400 | 2/45/37/8 | 6 | 71/21/19 | 40 / .6396 | 65/27/43 | 70 / .4815 |
| local_only | 110/400 | 0/57/36/17 | 6 | 74/36/16 | 52 / .5873 | 70/40/44 | 84 / .4545 |

全八task（global1/3/11/13/23/26/31/32）：none为0/2/44/1/0/37/5/3，local_only为0/0/39/18/1/35/14/3。直接none→local_only为R/G/L68/42/24、churn66、J=.5075。相对上一版first-query-only，local_only200为+4、400为+7，400 R/G/L75/35/28、churn63/J=.5435；none400为−11、72/20/31。上述比较均保留原始成功集合，不能只报净增。

BBQ是持续丢失的主要贡献项：none23→1，原23成功全部丢失，新增1；local_only34→18，保留12、新增6、丢失22。扣除BBQ，其它7task分别85→91、80→92；总分下降并非所有validation任务同步退化。这些新臂尚未补轨迹，因此不能把原contextual回放中的绿色干扰瓶选择，直接当成它们的已观察失败行为。

**当前裁决：** local_only相对all在两个固定validation节点均提高（+11/+20），相对none也均提高（+6/+18）。这支持“保留local上下文、联合去掉两处额外检索条件”在本学习实现下改善总分，不能再把冻结三处零40/96用于否认学习简化的价值。none的训练任务39→57，但validation108→92、BBQ23→1，故全部删除未修复获取/迁移分歧。local_only也仍有BBQ丢失、Spatial归零和高churn，不能称整体能力保持已恢复，更不能把联合效应归给其中单一路径或当作v5.2/v6分差的主要解释。

完整逐task、suite、同节点及相邻成功集合见`language/{arm}_step{200,400}/{validation,train96}_analysis/`，总索引`language/completed_language_matrix.json`。两臂各两个checkpoint、8个新闭环面板共1984行全部保留；本轮没有Test、最终视频controls或held梯度。

## 下一项：分开检验两处额外检索条件（结果前登记）

首轮的正效应是两个入口的联合删除。下一项补足local固定为开启时的2×2检索条件矩阵，复用已经完成的all与local_only，只新增以下两个fresh学习臂：

| 臂 | local条件 | H-read条件 | Compiler首次额外条件 |
|---|---|---|---|
| all（已有） | 开 | 开 | 开 |
| local_h_read（新） | 开 | 开 | 关 |
| local_compiler（新） | 开 | 关 | 开 |
| local_only（已有） | 开 | 关 | 关 |

local_h_read相对all只关闭Compiler入口；local_compiler相对all只关闭H-read的条件输入，保留其可学习bias。两臂也分别相对local_only只恢复一处入口。共同参数声明与初始化顺序、local上下文、真实原生prefix/视觉/50H、过程、decoder、Action Meta和纯FM协议均保持，不改任务或查询权重。新枚举只进入隔离探索运行面，候选行为不合入main。

此处没有重做first-query-only：它曾删除Compiler后续残差中的重复语言，但保留首次query条件；本次变量是是否提供这个首次额外条件及H-read条件。也没有重做冻模lesion：两个新增臂均从相同fresh初始化真实学习。已有Target-Owned、LPCP、Unified等共享或读取架构不参与本矩阵；不因GPU释放就启动它们或恢复v6。

预注册判据：若关闭H-read在Compiler开/关两种背景下均改善相邻闭环，才支持H-read条件的可重复负作用；Compiler同理。若收益只在联合删除出现，则保留交互解释，不能分别给两入口定责。若单臂只提高训练任务，或验证集只有一个节点更好、广度/保持明显变差，不将其当迁移修正。若某单臂保住Spatial且保留local_only的Object/Long收益，则是需要进一步独立teacher/state复核的具体候选；不自动正式采纳。

各新臂仍fresh400updates，固定200/400完整checkpoint、validation400和held-video train96、独立3072query动作诊断；同seed7、teacher0–15/query16–41、4suite×64query、LR3e-5/warmup8及正常重采样FM time/noise。实际曝光继续逐项核对，允许相同合同下正常BF16与物理微批低位差异，不按loss选点或补500。先做真实短profile验证开启/关闭入口及其余完整图梯度，再按实时设备、quota和精确输出预算登记launch；本段登记不是资源检查的替代。

## 检索分离：200步完整闭环结果

两个新增train96均完整exit0、全部worker返回0，与all和local_only实际strict配对通过。每task仍是固定4个独立state、teacher46–49；该面板描述训练任务的闭环行为，不能替代validation400或相邻保持。

| 臂（H-read/Compiler） | 成功 | S/O/G/L | breadth | 相对all R/G/L | churn / J | 相对local_only R/G/L | churn / J |
|---|---:|---|---:|---|---|---|---|
| all（开/开） | 41/96 | 11/11/13/6 | 18 | — | — | — | — |
| local_h_read（开/关） | 40/96 | 14/9/12/5 | 17 | 29/11/12 | 23 / .5577 | 34/6/12 | 18 / .6538 |
| local_compiler（关/开） | 43/96 | 10/12/14/7 | 18 | 38/5/3 | 8 / .8261 | 32/11/14 | 25 / .5614 |
| local_only（关/关） | 46/96 | 13/14/12/7 | 18 | 32/14/9 | 23 / .5818 | — | — |

在这个训练节点，关闭H-read条件在Compiler开/关的两个背景分别净增2/6；关闭Compiler条件在H-read开/关时分别净−1/+3。两处联合关闭相对all的+5，没有由任一单独关闭完整重现。Compiler单入口与all保留38个共同成功，变化较小；H-read单入口改善Spatial但在其它suite有损失。以上是单节点、单seed的实测条件效应，不能据此宣布稳定交互或将主因定责到某个入口。这些训练任务结果须与下文validation200和待完成的400/相邻证据一起解释。

全部逐task、suite、成功集合与配对原件为`runs/analysis/horizon_relation_writer_20260908/causal_learning_20260909/retrieval/{arm}_step200/train96_analysis/`，其中`completed_summary.json`、`vs_contextual_same_step.json`与`vs_local_only_same_step.json`保留完整依据。期间观察到共享存储RPC/文件读取等待；保留原进程后两面板正常结束、全部96行和输入配对通过，未重启、补行或调整超时来改变结果。墙钟包含该等待，不直接用来归因模型吞吐。

两个新增validation400现也完整exit0、全部worker返回0，实际strict配对通过；检索分离200的四个新面板共992行已完整保留。

| 臂（H-read/Compiler） | validation成功 | S/O/G/L | breadth | 相对all R/G/L | churn / J | 相对local_only R/G/L | churn / J |
|---|---:|---|---:|---|---|---|---|
| all（开/开） | 103/400 | 1/54/37/11 | 7 | — | — | — | — |
| local_h_read（开/关） | 108/400 | 3/57/34/14 | 6 | 75/33/28 | 61 / .5515 | 85/23/29 | 52 / .6204 |
| local_compiler（关/开） | 98/400 | 4/52/38/4 | 7 | 77/21/26 | 47 / .6210 | 72/26/42 | 68 / .5143 |
| local_only（关/关） | 114/400 | 3/62/36/13 | 6 | 77/37/26 | 63 / .5500 | — | — |

两个新增臂的validation global1/3/11/13/23/26/31/32分别为H-read单入口0/3/28/29/0/34/12/2、Compiler单入口1/3/31/21/0/38/2/2。它们没有恢复弱Goal任务；Spatial仍很低，breadth中的单次成功不代表能力修复。

**200节点区分到的条件效应：** 关闭H-read在Compiler开/关背景下分别净−5/+6，与train96两背景都为小幅正增不同；因此不能把H-read条件笼统判为有害。关闭Compiler在H-read开/关背景下分别净+5/+16，但其train96效果为−1/+3。联合关闭在200仍最好，任一单独关闭均未同时重现联合方案的验证总分和训练任务结果。以上支持入口之间的作用依赖及训练任务/未见任务的不同响应，不证明已解释Horizon相对旧模型的主要分差。Compiler简化的验证收益是否保持，仍由预注册400及相邻成功集合裁决；不按这个200节点提前正式采纳。

检索分离200总索引为`retrieval/step200_completed_matrix.json`；新增验证的完整原件在各`retrieval/{arm}_step200/validation_analysis/`。Compiler两节点的完整结果见下节；当时H-read继续原训练到400，最终结果见末节。

## 检索分离：Compiler单入口400完整结果

local_compiler的200/400四个面板均完整exit0，全部worker返回0；400与all、local_only及自身200的actual strict配对通过。两checkpoint完整学习状态、累计1600条件/102400queries和固定held3072输入均已检查。

| 面板 | 200→400 | 400 S/O/G/L | breadth200→400 | 相对all400 R/G/L | churn / J | 相对local_only400 R/G/L | 相对自身200 R/G/L |
|---|---|---|---|---|---|---|---|
| train96 | 43→65 | 17/18/18/12 | 18→23 | 43/22/6 | 28 / .6056 | 47/18/9 | 37/28/6 |
| validation400 | 98→90 | 0/45/34/11 | 7→5 | 76/14/14 | 28 / .7308 | 73/17/37 | 58/32/40 |

train96相对local_only的churn27/J=.6351，相对自身200为34/.5211；validation相对local_only为54/.5748，相对自身200为72/.4462。400 validation global1/3/11/13/23/26/31/32为0/0/43/2/0/34/10/1。BBQ21→2，R/G/L0/2/21，原21个成功全部丢失；其余七task合计77→88。没有新增轨迹回放，不能把此前contextual的绿色干扰瓶行为当成这一臂的已观察轨迹。

相对上一版first-query-only的400，Compiler单入口train96为59→65、R/G/L51/14/8、churn22/J=.6986；validation却为103→90、76/14/27、churn41/J=.6496。故65是训练任务行为的实际改善，不能以之替代迁移改善。原件为各400分析目录的`vs_first_query_same_step.json`。

这组受控学习中，关闭H-read在Compiler开启背景下使400训练任务比all多16次成功，覆盖扩到23task，但validation与all同为90且breadth更低，未修复BBQ保持。它进一步排除了“只要关闭H-read、训练任务更好就会解决未见任务退步”的具体修正假设；不排除该入口在其它背景的条件作用，也不能由训练成功证明整个表示/输出架构没有问题。相比Compiler单入口，关闭Compiler的local_only在两个validation节点分别多16/20，但400训练任务少9；这个入口的效果随任务分布和学习节点变化。

完整原件位于`retrieval/local_compiler_step400/{validation,train96}_analysis/`。H-read臂400随后按原注册完成，统一2×2及相邻保持裁决见后文；没有额外500续训、Test或最终视频controls。

## 检索分离：H-read单入口400训练任务结果

local_h_read的400 train96已完整exit0、三个worker均0，同节点与自身200的actual strict配对通过。成功40→56，S/O/G/L为18/17/16/5，breadth17→20。相对all400为R/G/L39/17/10、churn27/J=.5909；相对local_only400为46/10/10、churn20/J=.6970；相对自身200为31/25/9、churn34/J=.4769。相对first-query400的59为41/15/18、churn33/J=.5541，仍未超过该旧版训练总分。

因此在400训练节点，关闭H-read条件在Compiler开/关背景分别净+16/0；关闭Compiler条件在H-read开/关背景分别净+7/−9。入口作用明显依赖另一入口及学习节点，不能把简化普遍等同为更容易学习。H-read单入口Long总数5→5，但只保留1、新增4、丢失4；总数不变不等于能力保持。该训练面板先于validation400完成，不能据train96替代未见任务结果；最终验证证据见后节。

原件`retrieval/local_h_read_step400/train96_analysis/`包含逐task/suite、全部比较和worker返回码。两个新臂没有继续训练或补额外节点；完整2×2的最终裁决见后节。


## 检索分离全部完成：Compiler条件是已识别的有害因素，H-read不能一并定责

最后的local_h_read validation400完整exit0、9个worker均0，actual strict配对通过。其200→400为108→126，400 S/O/G/L为1/70/36/19、breadth6→7；global1/3/11/13/23/26/31/32为0/1/38/32/1/35/18/1。相对all400为R/G/L74/52/16、churn68/J=.5211；相对local_only400为89/37/21、churn58/J=.6054；相对自身200为82/44/26、churn70/J=.5395。相对first-query400的103为80/46/23、churn69/J=.5369。

BBQ从29升至32，保留23、新增9、丢失6；其余七task79→94。此处终于出现训练任务40→56、未见任务108→126同时改善，并保住多数BBQ旧成功的具体学习修正。它否定了把当前后期BBQ崩落当成纯FM或整个当前前端的必然结果；没有改FM、数据、输出共享或原生读取可学习性就能改变该结果。但它仍未获得Spatial及部分Goal/Long能力，且其它成功有更替，不能宣布整体保持问题已解决。

| 关闭条件的作用 | validation200 | validation400 | train200 | train400 |
|---|---:|---:|---:|---:|
| 关闭Compiler，H-read开启 | +5 | +36 | −1 | +7 |
| 关闭Compiler，H-read关闭 | +16 | +20 | +3 | −9 |
| 关闭H-read，Compiler开启 | −5 | 0 | +2 | +16 |
| 关闭H-read，Compiler关闭 | +6 | −16 | +6 | 0 |

Compiler关闭在两种H-read背景、两个验证节点均正增，是当前共同初始化与学习协议下的方向一致证据；H-read关闭没有这种方向一致性。联合删除200较好，400却低于只删除Compiler，因此不能沿首轮结果继续断言联合删除最好。训练任务上的条件效应与验证不同，说明该入口在当前分布上可能帮助拟合而妨碍部分迁移；这是从实测差异作出的解释，尚未定位内部究竟形成何种错误编码或证明唯一优化机制。精确干预是将query_language(language)整条仿射输出置零，包含其bias；它定位该额外查询支路的联合学习效应，不能进一步把全部收益单独归因于纯文本内容、bias或参数化中的某一项。target/rank learned queries仍保留。

本轮两个新增学习臂、四checkpoint、八新闭环面板共1984行全部完成，完整逐task/suite、四条条件对比及相邻成功集合索引为`retrieval/completed_retrieval_matrix.json`。与首轮语言对照合计3968个新闭环行，均无Test、held梯度、最终时序controls或正式方法采纳。运行数量不构成goal完成；下一步改为复核这个实质效应，而不是立即启动已准备的共享或VL架构候选。

## 候选复核：在结果前冻结下一项合同

待复核候选固定为local_h_read：保留local逐帧上下文与H-read条件，只关闭Compiler首次额外文本条件。基线固定为all。两者的后续Compiler纯语言残差均已按first-query-only移除；本轮比较不把原生exact language、视觉上下文或H-read一起删除。

1. **另一次初始化的成对学习。** 两臂optimization seed固定改为11，data seed仍7，teacher/query序列、FM抽样、累计曝光、源模型、数据池、4×64、LR3e-5/warmup8及所有模块保持。只做这一对独立初始化，fresh400、固定200/400 validation400与held-video train96及held3072；不试多个seed挑最好。沿用clean pushed隔离运行面45e16633，不改canonical源码。两个seed的基线必须各自配对，不把seed11候选对seed7基线冒充同初始化因果效应。
2. **冻结seed7两节点的另一套正确视频配对。** 固定新的无放回视频schedule seed20260910，对all/local_h_read各200/400重新分配视频到官方50个初始状态，仍为correct视频、K1、相同env/policy RNG。每task已物化全部50视频，因此复用相同checkpoint的原LoRA，不重新训练、融合或挑video。这检验换教师与状态配对后的效应，不声称是新任务、新视频资产或新的初始化池；official validation只有现有50个state。该探索复核不作正式checkpoint选择，不运行wrong/shuffled/reversed/no-video或Test。
3. **必要的行为定位。** 复用已登记的BBQ四state0/12/25/37，为候选200/400补相同correct回放，检查是否实际恢复目标对象选择，不能把原contextual绿色瓶轨迹直接套到候选。只描述这些预先选定实例；完整400负责效应裁决。若独立配对或初始化不保留收益，报告依赖条件并回到尚未区分的主要解释，不以seed7的126单点宣布修复。

复核支持标准：Compiler关闭在独立初始化/视频配对中仍表现为相邻验证方向改善或避免后期崩落，且BBQ保留与Object/Long收益可复现；同时报告所有task、breadth、R/G/L/churn和弱suite。如果只在原seed/配对有效、改善来自另一组不相干成功或伴随同量丢失，则将可靠性限定到原实验，不能正式采纳。没有预设必须达到145或继续500的要求。

两组学习及其200/400 banks预计新增峰值40GiB；换配对banks通过已验证的同checkpoint硬链接复用，回放/原行及余量计入该预算。真实launch仍以当时双节点GPU、quota和峰值检查为准。共享rank与可学习VL候选继续保持准备/只读审计状态，旧v6重训继续暂停。

### 候选BBQ行为复核已完成

原四state0/12/25/37的候选200/400共8条回放完整完成，四worker均exit0，实际task/state/video/adapter、env/policy RNG配对与终态谓词检查通过。两节点这次均3/4；200四例的成功标签均与历史相同，400的state12由历史成功变成本次失败（其它三例仍成功）。多例执行步数也有变化；输入合同配对通过不等于逐步轨迹或成功必然完全相同，分叉成因尚未定位，未重跑追成功或改变数值模式。

双相机检查显示：候选400四例都围绕正确的棕色橙盖BBQ操作，states0/25/37抓取并送入篮子，state12反复尝试但未完成运输。此前同四state的all400均操作绿色干扰瓶。因此本次实例支持Compiler简化伴随目标选择保持的改善，不能把改善全部归为一般抓取/运输技巧。200的state12也未送入篮子，过程中碰倒绿色干扰瓶并移动BBQ；不把所有失败都统一成目标识别问题。

这是事后选定task、沿用固定state的描述性回放，不替代完整400或新初始化/换视频复核。原件`compiler_confirmation/replay/completed_summary.json`、双相机contact与轨迹全部保留。初次缺少LIBERO配置链接的启动失败发生在任何queue claim/rollout之前，原日志另存`bootstrap_attempt1`；补原有配置symlink后的这次完成不属于结果驱动重试。

### 历史分差与尚未取得的能力须分别解释

为判断当前修正覆盖了哪些差距，复算三个此前已明确引用的历史节点原始400行，并与当前seed7 all/LH400分解到suite。以下是历史配方的描述性参照，未把不同训练协议、视频分配或初始化强行当成受控配对；不能用36除以历史分差，声称已经解释某个百分比。

| 固定参照节点 | 总成功 | Spatial | Object | Goal | Long |
|---|---:|---:|---:|---:|---:|
| v5.2 old900 | 132 | 19 | 63 | 39 | 11 |
| v6 old500 | 121 | 2 | 60 | 41 | 18 |
| v6 task-complete400 | 143 | 3 | 83 | 36 | 21 |
| 当前all400 | 90 | 2 | 44 | 35 | 9 |
| 当前local_h_read400 | 126 | 1 | 70 | 36 | 19 |

当前真正受控的Compiler关闭+36分解为Spatial−1、Object+26、Goal+1、Long+10；主要是Object/Long能力的获取或保持改变。它使本次候选总分高于旧v6 old参照，但不证明两个方法等价或候选已经稳定。对v6 task-complete143，候选剩余17的描述性差额主要为Object13、另有Spatial2与Long2；对v5.2 old132，候选Spatial少18、Goal少3，被Object多7与Long多8部分抵消。因此不能将历史差距笼统说成所有suite均变弱，也不能由总分接近忽略v5.2曾取得的Spatial task3能力（19/50，对当前候选1/50）。

另外，Goal23和Long32的严重弱势不是当前Horizon独有：上述历史节点Goal23为1/0/0、Long32为0/0/1；三个历史配方的已保存完整曲线中，Goal23最多1/50，Long32最多5/4/4。它们仍是EMBER必须解决的科学能力缺口，但不能当作“旧强模型已解决、被当前架构全部丢掉”的事实。Spatial task1在三条旧曲线也最多3/3/2；Spatial task3则存在v5.2的实质反例，不能一并归为从未取得。

原件`compiler_confirmation/historical_residual_context.json`包含当前和历史逐task/suite、原始行复算及历史曲线范围。独立初始化与换视频复核仍在运行；这些分解用于明确剩余问题，不提前采纳候选或启动未登记的新架构。

### 换正确视频配对：all400基线完成

固定schedule seed20260910的all400完整82/400，S/O/G/L2/41/35/4、breadth5，global1/3/11/13/23/26/31/32为0/2/40/1/0/35/4/0。三个worker均exit0，实际400个唯一task/state与完整contract一致，墙钟3320.17秒。旧视频分配下该checkpoint为90/400，两次BBQ都仅1/50；这里尚不比较候选净效应，也不将不同视频分配当作同视频strict配对。

原件`compiler_confirmation/reassignment/all_step400/completed_summary.json`及其`evaluation/`。候选LH400继续原评测；all200已在释放的GPU上启动，两个200面板仍须完整执行。新评测资源准入修正及首次LH预worker拒绝保留在progress/原件，不改变既定科学输入或本基线运行面。

### 换视频400配对完成：收益保留，但弱任务仍未修复

同一新的正确视频关联下，all82/400、local_h_read126/400，实际strict配对通过，五个worker均exit0。候选相对基线R/G/L为70/56/12、churn68、J=.507246；S/O/G/L为0/71/37/18、breadth4，对基线2/41/35/4、breadth5。逐global1/3/11/13/23/26/31/32为0/0/40/31/0/37/18/0。BBQ为1→31、保留1/新增30/丢失0；双物体Long为4→18、3/15/1。候选4197.07秒完整结束。

原视频关联的受控净增+36、这次+44，说明seed7的Compiler关闭收益并非只存在于原teacher/state关联；BBQ和双物体Long的末节点能力也在新关联下保留。但候选Spatial、弱Goal与另一Long仍零，原breadth7中的稀疏成功没有在新关联重现，不能以总数仍126代替广度或稳定性。独立初始化尚须由其自身all/LH闭环比较裁决。

all的新关联200也已完整99/400，S/O/G/L2/51/33/13、breadth6。其200→400为R/G/L53/29/46、churn75/J=.414063；BBQ23→1，仅保留1、无新增、丢22；Long13→4则原13全丢、另增4。LH200按原登记继续执行，尚不能计算候选在新关联下的相邻保持。原件`compiler_confirmation/reassignment_validation_step400_analysis/`含逐task/suite和全部strict比较。

### init11实际学习完成，闭环验证继续

all/local_h_read两臂已各完成fresh400且exit0，200/400完整checkpoint通过公开检查，各自累计1600条件/102400queries、各suite400，实际teacher/query/frame/FM RNG与data7参照一致；无Test或held梯度。all/LH的held3072均值200为.112488689/.112699040，400为.105458612/.106606225；这些值不负责判断候选行为。完整更新用时11608.09/15573.63秒，不能把共享GPU上的墙钟差直接归为架构吞吐。原件为`compiler_confirmation/{arm}_init11_step{200,400}/learning_summary.json`与checkpoint inspection。

all200与LH200各496条件物化已完整exit0、公开inspector与实际task/state/video映射检查通过，两组validation400已启动；all400的496条件物化也已完整exit0并通过检查；LH400随后已在换视频评测释放的gpu01p5启动物化，继续完成第四组及其余预登记闭环面板。评测仍按各自init11成对比较，不能把seed11候选对seed7基线冒充同初始化干预。动作拟合的同init11成对汇总另存`compiler_confirmation/init11_paired_learning_summary.json`，200/400候选分别5/14个task误差更低、总均值均稍高；该诊断不代替闭环。


### 换正确视频复核全部完成：保持收益与剩余丢失同时存在

最后的LH200完整101/400、三worker均exit0，S/O/G/L5/48/36/12、breadth6，global1/3/11/13/23/26/31/32为0/5/22/26/0/36/8/4，墙钟3467.19秒。四个换视频面板共1600行已完整结束，11个worker均0；200同节点、400同节点及两臂各自相邻的actual strict配对全部通过。仍是同一官方50个state池与相同视频资产的另一种无放回关联，没有新增独立初始化池。

| 新视频关联 | 200→400 | 自身相邻R/G/L | churn / J | breadth200→400 | BBQ200→400及R/G/L |
|---|---|---|---|---|---|
| all | 99→82 | 53/29/46 | 75 / .4141 | 6→5 | 23→1，1/0/22 |
| local_h_read | 101→126 | 71/55/30 | 85 / .4551 | 6→4 | 26→31，17/14/9 |

候选对基线200为99→101、R/G/L64/37/35、churn72/J=.4706，早期净差仅+2；400为82→126、70/56/12、churn68/J=.5072。它在这一视频关联下仍避免BBQ后期近乎全失，并继续取得Object/双物体Long能力；不能把两节点统称为同幅度稳定优势。

候选自身相邻保留71、新增55、丢失30，churn85反而高于基线75，说明更高总分仍包含明显行为更替。BBQ保留17/26，弱于原关联的23/29，但远多于同关联基线的1/23；这支持部分保持改善，不是零遗忘。Spatial task3的5个成功全部丢失，Long moka任务4→0、四个旧成功全失，最终breadth由6降至4。双物体Long8→18仅保留4、新增14、丢4；Goal36→37也有30/7/6的更替。不能用总分增长或两个400都为126掩盖这些缺口。

完整原件为`compiler_confirmation/reassignment_validation_step{200,400}_analysis/completed_summary.json`及各自paired比较JSON。独立初始化行为仍由当前两臂自身的完整200/400面板裁决，未通过新视频配对替代它。最新Owner已恢复分析后的复核停止点：本结果不触发正式方法采纳或下一次正式训练。


### 独立初始化200完成：早期小幅净增，不能外推后期保持

init11的all与local_h_read两个200面板各完整400行，六worker均exit0，实际输入/状态/视频及env-policy RNG严格配对通过；all3500.67秒、候选3627.25秒。四组200/400、validation400+train96 LoRA banks此时均已完整生成并通过公开inspector与实际映射检查。

| init11，step200 | correct | S/O/G/L | breadth | 相对all R/G/L | churn / J |
|---|---:|---|---:|---|---|
| all | 100/400 | 1/57/31/11 | 6 | — | — |
| local_h_read | 104/400 | 2/63/32/7 | 7 | 76/28/24 | 52 / .59375 |

all按global1/3/11/13/23/26/31/32为0/1/30/27/0/31/10/1，候选为0/2/28/35/1/31/5/2。BBQ27→35、保留25/新增10/丢2；双物体Long10→5则仅保留2/新增3/丢8。此处候选早期总分净+4、Object+6，但Long−4；breadth多出的Goal task23只出现一次成功，不能称广泛获取。它与seed7早期小幅净差方向相同，却不证明独立初始化下已避免后期BBQ崩落或保留Long收益。

完整原件`compiler_confirmation/init11_validation_step200_analysis/completed_summary.json`及`candidate_vs_all.json`。两组400验证和预登记train96继续；400将分别与各自init11的200比较保持，不与seed7基线混作同初始化干预。当前仅报告这个已完成节点，不正式采纳方法。


### 独立初始化200训练任务配对完成

同一init11的200 train96为all41、local_h_read44，六worker均exit0、actual strict配对通过；S/O/G/L为13/8/14/6与14/12/13/5，breadth均17。候选保留37/新增7/丢失4，churn11、J=.770833；Object保留全部8个基线成功并新增4个，Spatial净+1，Goal/Long各−1。两个面板完整耗时932.49/928.98秒。

这个节点的训练任务净+3与validation净+4都较小，任务覆盖未增加，不能由此宣布稳定泛化或保持修复。两臂400及各自200→400仍待完整比较。原件`compiler_confirmation/init11_train96_step200_analysis/completed_summary.json`与paired JSON；all400 train96已接续，最后LH400 train96仍按预登记待执行。


### 独立初始化基线400与相邻证据完成

init11 all400 validation108/400、train56/96均完整，六worker及outer exit均0；两面板分别3297.31/870.09秒。实际200→400 strict配对通过。validation S/O/G/L从1/57/31/11到3/46/40/19、breadth均6，相邻保留62/新增46/丢38、churn84/J=.424658；train从41到56、breadth17→20、相邻30/26/11、churn37/J=.447761，400 S/O/G/L16/18/13/9。

基线BBQ27→5，只保留4、新增1、丢23；其它七task73→103。因而另一初始化重现了BBQ大量丢失，却没有重现验证总分下降：其它task的获取抵消了该退步。必须把局部保持与整体总分分开，不能将seed7的总分走势外推给所有初始化。此时local_h_read400两个面板仍在运行，不能提前宣布候选独立初始化有效。

完整原件`compiler_confirmation/all_init11_step400/{validation,train96}_200_to_400.json`及各面板完成汇总；两臂全完后仍按原合同出完整同节点及相邻矩阵。最后候选train96已接续刚释放的gpu01p5/6，每卡三worker，复用原动态队列和96条件；无需改变科学合同。


### 独立初始化验证全部完成：净增复现，保持修复未复现

init11 local_h_read400为119/400，S/O/G/L1/54/44/20、breadth6；三worker和outer exit均0，3441.03秒。两臂200/400四个验证面板全部完成、实际同节点与各自相邻strict配对通过。

| init11验证 | 200→400 | 相邻R/G/L | churn / J | breadth | BBQ200→400及R/G/L |
|---|---|---|---|---|---|
| all | 100→108 | 62/46/38 | 84 / .4247 | 6→6 | 27→5，4/1/23 |
| local_h_read | 104→119 | 67/52/37 | 89 / .4295 | 7→6 | 35→19，15/4/20 |

400同初始化all→候选为保留86/新增33/丢22、churn55/J=.609929，净+11；suite净变化S/O/G/L−2/+8/+4/+1。global1/3/11/13/23/26/31/32为all0/3/41/5/1/39/19/0、候选0/1/35/19/0/44/18/2。BBQ保留全部5个基线400成功并新增14，但另一Object task11少6，弱Spatial仍1/100、Goal23仍0，不能只报BBQ收益。

候选自身BBQ35→19丢20，双物体Long5→18仅保留1、另新增17/丢4；总分上升伴随大量成功更替。与seed7候选BBQ29→32、原成功保留23/29不同，init11只保留15/35。因此“关闭Compiler必能避免后期BBQ崩落/建立稳定保持”没有通过独立初始化复核；“该额外支路的删除能改善这两次初始化、两节点的验证总分”获得方向支持，400效应由seed7的+36缩至init11的+11，不能把初次幅度当固定疗效。这里没有多seed挑选或将总分净增替代原保持判据。

完整原件`compiler_confirmation/init11_validation_step400_analysis/{completed_summary,candidate_vs_all,all_200_to_400,local_h_read_200_to_400}.json`。最后候选400 train96仍待完成；全体确认结果之后再裁决下一项有区分力的原因实验，当前不正式采纳。


## 固定确认全部完成与最终裁决

最后init11候选400 train96为61/96、S/O/G/L19/18/15/9、breadth22，六worker和outer exit均0、458.72秒；与all56为R/G/L48/13/8、churn21/J=.695652。候选自身44→61、breadth17→22，保留36/新增25/丢8、churn33/J=.521739；Long原5个成功均保留并新增4，但训练任务的保持不能外推为validation的保持。

| 确认条件 | all validation200→400 | 候选validation200→400 | all train96 | 候选train96 |
|---|---|---|---|---|
| init7、原视频关联 | 103→90 | 108→126 | 41→49 | 40→56 |
| init7、新视频关联 | 99→82 | 101→126 | 不重复 | 不重复 |
| init11、原视频关联 | 100→108 | 104→119 | 41→56 | 44→61 |

预登记确认共12个完整行为面板3584行（init11八面板1984行、冻结init7换视频四面板1600行），全部worker0，实际同节点与各自相邻strict配对通过。另有预登记BBQ八次描述性回放，标签分叉及限制保留。总索引`compiler_confirmation/completed_confirmation_matrix.json`，最后train配对在`init11_train96_step400_analysis/`。全部确认进程已正常退出，没有待运行确认面板、额外500步、Test或最终controls。

本轮识别到Compiler额外仿射查询支路在当前协议下的局部负贡献：关闭后两次初始化、两个节点以及换视频关联的验证总分方向改善，未改FM或输出共享即可改变部分退步。但独立初始化只保留15/35个早期BBQ成功，整体仍丢37、Spatial1/100、Goal23为零，**稳定保持修复未通过复核**。该结果不支持正式采纳为完整方案，也没有解释当前参数生成可学习性与旧新训练配方的全部差距。

下一项固定为已有隔离候选的同target跨rank输出映射绑定，比较当前all基线：只将D[target,rank,side,native,256]绑定为D[target,1,side,native,256]，语言/local/H-read/Compiler、完整视频与native50H、全部前端、rank16、FM及实际学习曝光均不改。选择原all基线而非新造结合臂，是为了直接区分原架构的输出独立性因素；不把Compiler删除与共享同时作为一个变化。历史Target-Owned的99/76/86/68是必须保留的反证，本次价值仅在当前前端和合法固定协议下的单接口识别，不将共享宣称为新发明或预定疗效。

共同seed7、teacher0–15/query16–41、4×64、LR3e-5/warmup8、fresh400及固定200/400 validation400/train96/held3072沿用已准备候选合同。共享D天然改变rank梯度聚合及AdamW状态，结果只归于这个参数绑定整体，不拆称纯参数量或纯正则化因果。不得从独立D checkpoint转换启动，不额外挑seed/学习率或续500；先真实profile并依据当时双节点GPU、独立quota、峰值和精确命令启动，CPU准备不代替这些检查。具体接口、初始化与历史边界见隔离候选`docs/horizon_causal_sharing_candidate_20260909.md`。这是下一项原因实验，正式修改/采纳和正式训练仍等Owner复核。


### 输出共享原因实验的实际启动

共享候选四更新真实profile与93帧最长视频完整反向均通过，全部有效路径含Compiler语言与H-read仍有梯度、source参数冻结；平均30.0015秒/update、峰值34.8792GiB、最长15.1864秒/34.7127GiB。已有CPU绑定/梯度检查与实际配置比较只证明干预按合同发生，不证明能力改善。首次profile在model/data/update之前因缺失本地远端authority ref退出，fetch精确既有ref后原配置通过，日志保留`sharing/bootstrap_attempt1`。

随后按新的双节点GPU/独立quota记录，从clean pushed detached c63f55dc使用gpu01p5/6、world2/micro8,8 fresh0→400，固定200/400，init/data均7，输出`horizon_causal_within_target_rank_shared_seed7_20260910`。actual前4更新16条件1024queries与原all的task、teacher、action query/frame、RNG、task权重逐项配对；主方法未改，无held梯度、最终controls或Test。约3.3小时更新仅为profile估计，不能替代后续行为结果。所有精确合同/资源/实际进程和配对原件在`causal_learning_20260909/sharing/`；两个checkpoint的物化与闭环脚本已按自身schema准备，尚无模型能力结果。

## 结构与学习机制深化：目前究竟定位到了什么

### 净学习、保持和平台是三个不同判断

关闭Compiler额外query支路的候选，在两个初始化的200→400都取得了**真实的净学习进展**，不能把保持未修复写成“已经学不动”。init7 validation108→126、train40→56；init11 validation104→119、train44→61。候选验证自身相邻R/G/L分别82/44/26与67/52/37：新增超过丢失，因而总能力提高；同时已有成功明显丢失，因而不能宣布保持稳定。init7保留75.9%的早期成功、churn70/400；init11保留64.4%、churn89/400。两个初始化不是同一模型轨迹，不能拼成更多时间节点。

**尚未证明平台、已经穷尽或再训无效。** 这两个候选只有固定200/400完整行为节点；不能据此测出400附近斜率，更不能外推500之后。停止本轮后续完整训练来自Owner明确预算/授权上限，并非平台结论。正式>145、相邻稳定与弱task扩展仍未达标，也不能把“继续学习”换成“继续训练必然达标”。

现有四条完整400步日志的最后两个50步窗口（151–200→351–400）平均FM分别为：all init7 `.107737→.100437`、候选init7 `.107759→.100266`；all init11 `.108129→.099808`、候选init11 `.108324→.099759`。同类拟合均继续进步，而all init7验证退步、另三条净上升；因此这个均值不能解释不同策略的保持差异。四run全部1600条已记录全局梯度范数均未超过实际clip阈值1，最高分别`.4164/.5025/.9794/.6692`；**实际触发梯度裁剪不是这四臂差异的解释**。这不排除AdamW坐标预条件、梯度方向或不同模块间共同适应。

### Compiler：从支路因果证据推进到具体检索接口

实际实现为`horizon-causal-retrieval`提交45e16633的`writer/horizon.py`、`writer/attention.py`。每个target/rank槽的初始身份为

\[
q^{id}_{tr}=e_t+e_r,\qquad b_L=W_L\ell+b_L^0,
\]

第一Compiler block中每个head实际计算

\[
q_{tr}=W_Q\operatorname{LN}(q^{id}_{tr}+b_L)+b_Q,\quad
k_i=W_K[\operatorname{LN}(P_i)+p_i]+b_K,\quad
v_i=W_V\operatorname{LN}(P_i)+b_V,
\]
\[
\alpha_{tr,i}=\operatorname{softmax}_i(q_{tr}^{\top}k_i/\sqrt{32}+\pi_i).
\]

这里8head、每head32维；`p_i`由真实frame index/5编码，K1下`π_i=-log(T)`对所有帧相同，不能改变该视频内部的softmax分配。语言支路只加在第一block的lookup；该block的残差起点仍为`q_id`，后续还有cross输出、自注意力、FFN和第二Compiler block。P4始终提供真实视频Value，exact language也仍进入原生图文prefix及逐帧local/H-read。删除该支路不等于删除语言、视频Value或target/rank身份。

**新诊断支持：额外共用位移确实强烈压缩第一层检索query之间的差异。** 预登记CPU分析只读八个已有checkpoint各933,888参数、冻结source的77个token行，以及固定train24/validation8 exact language。复用实际language reader与原生task span、位置和embedding缩放，FP32计算；没有完整policy/视频forward、动作、优化器、梯度或Test。原件`causal_learning_20260909/mechanism/query_geometry.json`保留每task数值、所读checkpoint及实现，计算主体1.71秒。

定义query差异幅度为同一task内608个投影后query相对其均值的RMS向量距离；表中比值以**同一个all checkpoint、同一LayerNorm/Q投影权重、仅代数置零额外位移**为分母，不是新闭环分数。

| all节点 | validation语言位移/身份的中心化范数比 | query差异幅度剩余比例 | 同task query平均cosine | 仅保留仿射bias时cosine |
|---|---|---|---|---|
| init7 / 200 | 26.10 | 3.85% | .99901 | .63025 |
| init7 / 400 | 28.30 | 3.55% | .99922 | .63019 |
| init11 / 200 | 21.72 | 4.63% | .99853 | .66192 |
| init11 / 400 | 23.66 | 4.25% | .99894 | .66141 |

train24比值同样为3.53%–4.58%，不是只在validation语言上出现。移除支路的候选实际query平均cosine约`.044–.055`；all自身置零反事实约`.040–.049`。这定位于**加法位移→LayerNorm→第一层Q投影**，超过了此前仅凭支路名称猜测“语言抢占视频”的证据强度。

数学上，LayerNorm的局部Jacobian近似为`diag(γ)/σ`乘去均值和去径向分量的投影。当共同位移主导特征方差时，较小的身份差异会被约`1/σ`缩放，因此归一化并不自动保持不同槽的检索差异。实际位移范数约10–13、显式affine bias范数约.59；跨32条语言的位移散布只约为平均范数的6.6%–7.9%。故当前证据更接近“**以共同方向为主、仍含task差异的语言读出压过小身份向量**”，不能写成“每个task特有语言信号都很强”，也不是仅`.query_language.bias`本身造成的常量偏置。纯bias与语言读出共同分量尚未通过行为学习对照分开。

同样值得区分的是漂移：all的200→400实际Q逐元素RMS漂移为init7 `.2161`、init11 `.2698`；候选为`.0287/.0308`。32条静态语言读出本身的validation RMS漂移为all `.2945/.2666`，候选`.0314/.0227`。语言reader在静态Compiler输入与逐帧上下文读取之间共享；删除Compiler支路也删除了它对这个共享reader的一路梯度，因而**检索尺度、查询漂移和共享reader共同适应同时被改变**。现有证据没有独立测量这些梯度是否冲突，不能把一种具体优化解释宣布为唯一根因。

**仍未验证的关键环节：** 相似Q不等于softmax饱和。是否只读少数帧、不同槽是否真有相同attention、读出的证据是否错误，还取决于真实K/V。`diag(α)-ααᵀ`会影响Q/K梯度，而Value梯度仍按α传递；没有实际attention证据就不能声称梯度消失。身份残差、后续block和独立D也能补偿相似检索；这里不能推出最终C、A/B或LoRA rank已经坍缩。闭环fresh消融支持支路的局部负贡献，CPU几何支持一个具体中间机制；二者尚未构成“query压缩独自导致全部分差”的完整中介因果证明。

### 输出D绑定：函数空间与学习更新究竟改了什么

实际提交c63f55dc的`writer/native_factor.py`只绑定同target、同side的跨rank D，A/B两侧和38个target之间仍独立。令`H_A/H_B`为16个槽经共同U和GELU后的16×256代码，则共享版为

\[
A=A_0+H_AD_A^\top,\qquad B=D_BH_B^\top,
\]
\[
\Delta W=sBA=sD_BH_B^\top A_0+sD_BH_B^\top H_AD_A^\top.
\]

固定checkpoint后，所有条件的B列均在同一`col(D_B)`中；A的**残差**行均在同一`col(D_A)`中，整个A还包含A0。独立版本每个rank各有自己的字典。绑定改变的是跨槽共享的方向字典和函数空间；仍生成16个不同槽，`rank(BA)≤16`保留，既不强制rank1，也不保证实际rank16。native维度≤256的一侧未必进一步收紧子空间维数，但共同映射的绑定仍然存在。

这一点与Compiler分析存在可解释的联系：如果后续代码仍很相近，独立D可以把共同代码映成不同rank方向；绑定D后，这种补偿更依赖代码本身的差异。因此“共同检索+绑定字典”可能更难保持因子分工；反过来，共同字典也可能让兼容任务复用方向、减少独立漂移。**这两种都是待区分解释，不是已测交互**。本轮共享仍保留all的Compiler支路，没有删除Compiler×共享的交叉学习臂，不外推其组合结果。

对任一side的槽输出`f_r=D h_r`，绑定后的损失梯度为

\[
\nabla_D L=\sum_r g_rh_r^\top.
\]

独立版各自更新`D_r`。跨任务共享原本已经存在，新增的是同target/side的rank贡献进入同一参数和AdamW状态。总梯度平方`(Σ_rG_r)²`含跨rank交叉项，因此“分别AdamW后平均”不等于绑定更新，也不能解释成学习率自动乘16。当前每task的FM LoRA梯度仍乘.25，四task归约后一次clip/step/scheduler，任务权重未改。

以固定代码的局部SGD近似帮助理解：`δf_s=-ηΣ_r g_r(h_rᵀh_s)`。相似代码的槽互相影响更强；兼容需求可共享统计信号，冲突需求也可能抵消或相互干扰。AdamW会再按坐标改变更新。源码中的全局clip还可能因绑定改变系数；这只是结构可能性，是否发生必须由本run完整梯度日志判断，不能把它预设为观察到的原因。

identity初始化的FM梯度也有明确顺序。`∇_A L=sBᵀG`、`∇_B L=sGAᵀ`；fresh时D_A=D_B=0、A=A0、B=0，所以第一反向只有D_B可以打开。D_B有效更新后，D_A、U_B、C与共同上游才可能获得梯度；D_A再更新后U_A才打开，通常对应第1/2/3次反向。不能把AdamW对非零参数的weight decay当FM梯度，也不能把四步profile写成所有模块从第一步就在学习。两种D参数化都有这个顺序，绑定改变的是首次有效B更新的聚合以及随后返回上游的梯度。

因此，参数329,515,008→20,594,688本身不证明过拟合缓解。共享可能改善复用，也可能牺牲表达、改变早期学习速度或加强任务干扰；它不提供能力保持保证。历史Target-Owned的99/76/86/68仍反对“共享普遍有效”，但不同前端、前投影和协议使其不能裁决当前单接口。本轮效应最终只归于**已执行的同target跨rank A/B两侧D绑定整体**，不扩大为全部参数共享、跨target共享或纯参数量因果。

### 共享200的完整训练行为与实际输出几何

200步held-video train96已完整32/96，对同节点all41/96为R/G/L24/8/17、churn25/J=.4898，breadth18→16；S/O/G/L为7/10/11/4，对基线11/11/13/6。两worker及实际启动wrapper均exit0、实际strict配对通过，评测墙钟1201.75秒。完整逐task与suite原件在`sharing/step200/train96_analysis/`。这支持该绑定的早期训练任务行为代价；当时validation待完成，随后结果见下段；400仍待原定完整面板，不由此单点宣布最终无效或平台。

200 validation400随后完整115/400，对all103为R/G/L77/38/26、churn64/J=.5461、breadth均7；S/O/G/L为3/61/38/13，对all1/54/37/11。global1/3/11/13/23/26/31/32为2/1/35/26/0/38/10/3；BBQ25→26只是同节点臂间变化，尚无本run自身保持结论。三个worker及wrapper均exit0、actual strict配对通过，墙钟3585.10秒；完整逐task/suite原件`sharing/step200/validation_analysis/`。这是一项早期验证净收益，与train96的下降方向相反，不能概括成整体获取/迁移都更差。

**因此下述近单方向输出不能直接定性为有害缺陷。** 同一绑定同时带来train下降与validation上升，支持参数约束改变拟合与迁移的权衡；但尚未独立识别是有效归纳偏置、不同学习速度、方向正则化还是其它共同适应。不能只看rank几何就要求恢复高秩，也不能只看115就宣布稳定泛化改善。400及自身相邻证据仍是既定裁决的一部分。

预登记输出几何选择每个train24/validation8 task的首条登记episode（train ordinal32、validation ordinal0），all与共享200共64套已有LoRA、每套全部38target；实际condition/video映射一致，无按成功、范数或结果选择。仅CPU读取生成的A/B，用薄QR和16×16矩阵得到BA的奇异值，没有生成native方阵、模型forward或梯度，5.08秒完成。原件`mechanism/output_geometry_step200.json`保留全部1216个条件-target/每臂数值。这里的能量指奇异值平方，stable rank为`||M||F²/||M||2²`，不等于代数rank。

| 200输出（按条件-target等权） | all train / val | 共享 train / val |
|---|---|---|
| B第一奇异方向能量占比 | 57.60% / 57.67% | 99.99924% / 99.99928% |
| BA第一奇异方向能量占比 | 76.99% / 77.17% | 99.99976% / 99.99977% |
| B列平均绝对cosine | .53859 / .53889 | .9999934 / .9999937 |
| B stable rank均值 | 1.9427 / 1.9377 | 1.0000076 / 1.0000072 |
| BA stable rank均值 | 1.4023 / 1.3976 | 1.0000024 / 1.0000023 |

共享的全部1216个B第一方向均超过99.9979%，所有BA超过99.9896%；all的B范围18.93%–89.62%。所以不是均值被几个异常target拉高。**形式rank16保留，但实际生成修正几乎只使用一个方向**，是这个固定节点和条件面板的直接诊断事实。这里的单方向按每个条件、每个target定义，不等于整套38-target非线性policy只有一个自由度，也不等于不同task输出同一LoRA或没有条件依赖。all的BA也偏低有效秩，但程度明显不同。未乘共同LoRA缩放的BA范数均值all为4.90/5.15、共享为5.39/5.62；不能把共享问题概括为所有修正幅度都被压小。因子cosine依赖生成规约，而BA的奇异能量不依赖A/B之间可逆换基；两者共同报告。

为区分字典本身与实际使用，进一步只读共享200的38个D_B（约40MiB）。D_B第一奇异方向能量均值65.02%，逐target38.54%–74.46%，stable rank均值1.5723；前4方向89.50%。1.53秒完成，原件`mechanism/dictionary_geometry_step200.json`。D_B自身并没有达到生成B的近乎单方向集中，**因此仅说“共享字典只能表达rank1”不符合实际参数**。更窄的定位是：代码经过U_B/GELU与D_B映射后，对字典的实际使用高度集中。尚不能由此分开C本身差异小、U_B/GELU压缩差异与代码选择D_B主方向的共同适应；也没有直接证明仅恢复B多方向就会修复闭环。

这组结果使“相似检索在独立D下能被不同映射补偿，而绑定后代码分工不足”的解释更有针对性，但没有跨Compiler删除×绑定的实际学习对照，因此仍是受诊断支持的竞争解释。也不能排除近rank1是当前学习阶段的现象；必须保留既定400的对应行为和几何。低有效秩本身不是优化目标，不据此添加rank/正交损失、改scale或追加训练。

### 监督与前端：已经知道的链条，以及尚不能定责的接口

当前FM监督的是执行policy在独立同task示范状态、噪声动作与flow time上的速度场。其对Writer的梯度可以写成`J_compileᵀ J_policyᵀ ∇_v L`；这条长链把动作误差分配到D、C、P4、完整H与Action Meta。源码采用真实LoRA梯度和Writer/native重放，没有通过示范action输入Writer，也没有把held结果用于梯度。图接通与非零梯度只说明链条存在，不说明每个阶段收到的方向足以学出闭环可用修正。

teacher与action query跨episode保证无法逐帧复制，也意味着损失没有直接指定“teacher哪一帧对应当前query的哪一事件”。每个视频编译出的同一LoRA必须解释该task许多独立状态。这个全局信用分配既可能促成可迁移任务表示，也可能主要学到足以降低平均动作误差的静态/局部相关性；现有最终视频controls尚封存，不能宣布动态过程已经成为必要信息。更多同task teacher曝光也不等于更多独立meta-task映射；24个训练task对未见关系/对象组合的约束仍有限，但没有匹配task数量干预，因此不能把“小数据”当已证实主因。

读取侧的明确结构边界是：vision/Gemma及其prefix Z/KV冻结，Action Meta和Writer可学习；语言/检索消融没有改变这点。冻结prefix仍然包含真实图文语义，不能说“没有语义输入”；它是否缺少可供后端提取的细粒度对象/关系信息，现有实验证据尚未定位。旧v6含Text/VL/Action三Meta但还同时改变其它结构和协议，它的优势不能独自归因VL可训练。若未来考虑读取侧学习，必须让直接Z路径与R经KV路径梯度都回到真实prefix、放弃跨更新缓存；否则只是安装参数而没有实现该机制。本轮不实施这项修改。

能力保持还有独立的目标函数边界：降低当前均匀任务分布的平均FM，没有约束先前成功集合在闭环中的不变性。一个共享参数更新可改善一批示范状态，却让原先可成功的闭环偏离接触或目标选择边界。这解释“新增与丢失可以并存”的数学可能性，但当前没有逐task梯度冲突、恢复状态覆盖或接触误差的隔离证据。BBQ的首个可见问题是所选实例的对象选择；Long36/38部分实例已选对第二对象却操作失败，说明不能把所有缺口都缩成同一种语义错误。旧模型同类FM曾更强，亦反对仅凭这些边界直接宣布必须换loss或RL。

### 建议如何针对机制，哪些仍只是未验证设计

**当前最有实测支持的是简化额外Compiler query支路。** 它保留原生语言、上下文读取和视频Value，同时移除已观察到的查询差异压缩及该支路对共享reader的梯度；两次初始化和换视频关联支持局部闭环收益。但它尚不能恢复Spatial/Goal弱能力或稳定保持，不能作为完整成功方案正式采纳。

如果未来Owner决定保留额外task-dependent检索，一个机制明确的候选是对身份query与语言项分别归一化、显式限制语言项尺度，再组合检索，从结构上避免共同位移决定全部LN方差。它针对的是已定位的尺度与槽差异问题；并不保证更好，因为共同读取可能也有作用、Key/Value和后续补偿尚未测清。此处没有选择具体gate/scale、添加loss或启动训练，更不能把该未测方案说成已验证改进。

对共享的建议须等待本轮完整结果。若有必要在未来分辨A/B两侧约束，单独绑定B、保留独立A可检验共同输出方向复用与输入残差限制的不同作用；它仍改变B侧AdamW，不能称纯减参对照。当前没有该方案的效力证据或新增训练授权。类似地，扩大meta-task、学习prefix或补闭环状态覆盖分别针对泛化约束、可学习证据与执行信用；它们尚未被本轮独立识别，不能当作接下来必须机械执行的清单。
