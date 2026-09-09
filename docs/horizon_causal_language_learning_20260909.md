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

## 200节点完整闭环证据（400节点尚待完成）

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

## 400节点训练任务证据（validation尚待完成）

两个train96均完整exit0、全部worker返回0；与contextual400及各自200的实际strict配对通过。none为57/96，local_only为56/96，基线为49/96；三者breadth均20。

| 臂 | S/O/G/L | 相对基线R/G/L | churn / J | 相对自身200 R/G/L | churn / J |
|---|---|---|---|---|---|
| none | 14/18/17/8 | 43/14/6 | 20 / .6825 | 35/22/4 | 26 / .5738 |
| local_only | 15/16/18/7 | 41/15/8 | 23 / .6406 | 36/20/10 | 30 / .5455 |

因此200的局部描述不能外推为none没有学习收益：到400，它从39升至57并扩大成功任务覆盖。local_only从46升至56，但200相对none的训练任务优势没有保持。这些结果支持额外条件的影响随学习进程变化，尚不支持local条件有稳定正效应或两处检索条件就是主要原因。独立动作FM均值仍由基线略优，而400训练任务闭环由两条简化臂更好，进一步说明该动作指标不能替代行为裁决；不由此宣布FM目标失效。

400直接none→local_only的train96比较为R/G/L45/11/12、churn23、J=.6618；总分接近仍伴随成功得失。原件`local_only_step400/train96_analysis/vs_none_same_step.json`保留实际配对和成功集合。

收益参照也需限定：上一版first-query-only的train96为200步46、400步59。补充实际strict比较后，400 first-query→none为R/G/L46/11/13，→local_only为44/12/15；因此57/56是相对contextual49的恢复，尚未超过上一版59，不能包装成解决了Horizon总体能力缺口。200对应比较为33/6/13和35/11/11。四个原件位于各`train96_analysis/vs_first_query_same_step.json`；这是已有行为参照，主因果对照仍以当前contextual为共同基线。

完整逐task、suite、同节点及相邻成功集合见`language/{arm}_step400/train96_analysis/`。400两个validation400仍在原进程执行，未读取局部分数；未见任务迁移与反复丢失任务的解释待完整结果。
