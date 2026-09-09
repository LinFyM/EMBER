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

下一步：完成真实profile、冻结并推送探索代码、登记现场资源后运行两个预注册fresh臂。profile/launch状态与精确命令在analysis原件和progress记录，未取得结果前不写改法有效。
