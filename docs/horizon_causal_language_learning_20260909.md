# 后端条件路径：fresh学习对照

2026-09-09，结果前登记。归属`horizon_causal_learning_plan_20260909.md`，只作原因分析，不正式采纳或启动新正式方法。

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
