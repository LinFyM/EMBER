# 同目标跨情境配对：在强 T 上检验学习关系

2026-10-08。Owner要求核实专家建议，并持续自主推进，不能在少量尝试失败后停止整个研究。
本稿经progress登记后为唯一active批次。它采纳第三轮专家意见中的单变量检验，不采纳第二轮同时更换
encoder、完整A/B头与初始化的组合。整批结果回main消费并继续研究；本批停止线只约束这项假说。

## 1. 目标、依据与未知

最终目标仍是exact language和action-hidden正确视频在rollout前一次生成完整单套LoRA，
在未见初始化取得稳定、广泛、明显超过强MT的能力。当前T2340的correct/other为161/150，MT300为153；
同一T的公共策略103，两套正确视频相对公共策略共同新增64、共同丢失14。存在有益条件作用，不能假定视频未被用到。
但T的相邻147/154/160/154/159、后继C seen改善而held停滞，以及固定donor接手0/90，
共同限制了“现有条件函数已经学得可迁移操作规则”的判断；不据此确定唯一根因。

当前具体假说是：同BDDL跨episode的FM仍可能允许教学情境与接收情境的相关性成为捷径；
同exact language、同完整目标下打散这种相关性，可能让同一编译参数在不同自身状态上更有用。
原有跨episode已改变随机初始化；本次新增的是三个不同BDDL情境间的联合配对，不冒称首次阻断轨迹复制。
九个任务已被source71和旧C95接触，未进入T原36；新增曝光与配对效应必须分离。

对固定query Q及目标y，平方FM满足
`E_V ||f(Q,G(L,V))-y||² = ||E_V f(Q,G(L,V))-y||² + E_V ||f-E_V f||²`。
跨情境采样在同目标条件下使不必要的视频依赖付出功能代价；它也可能只削弱视频，不能推出V胜过充分L。
有限采样仍遵守同task跨episode排除，不能把本实现称为所有episode的完整笛卡尔积或严格独立的经验风险。
“情境解耦已发生”“视频知识更可迁移”“大幅突破强MT”均是待验证判断，不由该恒等式证明。

完整原理保持T：双RGB/语言→固定原生probe读取X/H→P/C/D/O顺序写入M→完整(A,B₀+M)→自身hidden/动作。
教学侧无state/action；令h为RMSNorm(H)，每层`K=column_normalize(A X)`，
`Y=O{GELU(PK+Ch) ⊙ D(h[t+1]−h[t])}`，再以`M ← M+(Y−MK)Kᵀ/50`顺序汇入。
最终动作仍由自身图像/state/language与10步flow求得。
Product改变真实query回传给同一教学编译的梯度，不加latent一致性、公共loss、物理标签或部署Reader。
正面预测是完整跨情境控制与官方能力改善；仅FM下降、M变小或内部表示趋同不提升方法资格。

## 2. 两臂、数据与信息墙

两臂Within/Product从同一个完整T2340 ECP分叉：
`/data1/user/ymdai/ember_runs/operator_read_write_learning_20260928/continuation2340/T/train/attempts/continuation/checkpoints/macro_00002340`。
该ECP的模型、optimizer、scheduler、四rank RNG与sampler文件当前存在；实际恢复正确性由实验session核验。
继承参数、optimizer及学习率进度，不重新初始化、不换seed/架构/rank/归一化。这是有意改变训练分布的科学分叉，
不是原训练流exact-resume，也不是fresh Writer。原训练来源e2afbfd7及其完整合同保留，读取代码单独登记。

原36全保留：`0,1,2,4,5,7,12,13,14,15,17,19,20,21,22,25,28,29,32,34,35,36,37,38,42,43,51,55,56,62,64,73,95,96,97,101`。
只加下表9个source71任务，合计45；source-local ID加40为全局ID：

| 完整目标 | SCENE1 | SCENE2 | SCENE3 |
| --- | --- | --- | --- |
| book in front compartment | 113 (73) | 118 (78) | 121 (81) |
| book in left compartment | 114 (74) | 119 (79) | 122 (82) |
| book in right compartment | 115 (75) | 120 (80) | 123 (83) |

每列是情境，每行才是可交叉目标组。依据`configs/pi05_source_corpus_v1/source_manifest.json`及`overlap_audit.json`，
九项均active、每项50条成功示范，同组exact language及完整In目标一致，均非固定validation/test及其等价项。
执行者在实际资产上保存显式45-task allowlist、同组language/完整goal和source provenance，不以编号代替语义审计。
三情境主要改变书的起点/朝向与干扰物；caddy位置区间和朝向相同。SCENE3的contain命名初始region属于桌面，
初始谓词为On，不是书已在caddy中。不得称为移动容器、槽位程序或柜门/炉腔阶段连接数据。

冻结source1000、source normalization、目标coverage24/8/8与所有官方预处理沿用T。
教学输入只含exact language、同步双RGB，stride5、K1；不读动作/state、task ID、文件名、pose或结果。
query自己的RGB/state/action仅用于合法训练FM；同scene query不得等于该scene的teacher episode。
同组不同global task的episode编号相同不构成同episode。标签和元数据不得流入教学条件。
validation/test无梯度；不追加RL、纠错rollout标签、专家、语言baseline、static/wrong/时序controls或Test。

## 3. 匹配采样与216更新

每臂216次完整更新，每步4个条件、每条件28个query，共864条件、24,192 query使用。
新9项占216条件（各24次/672 query），原36占648条件（各18次/504 query）；分层权重在两臂完全相同。
所有宏步112个query等权、每condition权重1/4。不是45task全局等权，而是事前登记的3/4旧、1/4新分层。

新root seed20261008固定。宏步u=0..215：u mod3=0时放同目标三情境各一个teacher条件和一个旧条件；
其余宏步放四个旧条件。新目标按front/left/right循环，故每目标24个triplet。
旧648个事件沿T原task排列/teacher/query算法，从visit260起完整走18轮；两臂同一宏步同一旧事件。
新task的teacher由种子[root,1,global_task]对50条做固定置换，visit0..23各用一次；
query由[root,2,global_task,visit]从余下49 episode无放回取28，每episode在0..length−2取一个frame。
沿原offset1、50-action chunk和末动作补齐，不改FM horizon/通道权重。标签只读原canonical HDF。

每个新triplet先构造三组Q_c，各28条，连同真实tau/noise固定为84个query记录。
Within把Q_c全部交给同scene teacher V_c；Product把相同84条重新分配给三套V生成的LoRA。
Product的3×3分配矩阵每行/列和均28：每格9条，再每行/列各加1；额外1所在列按visit mod3轮换，
三个visit内每个scene pair均28条。每个teacher在同一更新中同时服务三种query scene，包含对角，非固定场景互换。
各Q_c先用[root,3,goal_index,visit,c]置换再切分，所有query在两臂同宏步各用一次。
24个triplet后每目标的9个scene pair各224条query使用；每teacher/query scene边际均672。

匹配对象必须是query的task/demo/frame、动作/state/RGB、tau和实际Gaussian noise；不能只说两个condition seed相同。
tau/noise先按query原身份生成再随query搬移，不因新的teacher归属或物理microbatch重新抽样。
教学视频边际、每macro query边际、更新顺序、loss权重和optimizer步数完全匹配；不平均视频或最终LoRA。
保存可读事件/分配清单、各cell计数及实际消费者配对核验，不能只验证计划生成器。

沿T原全部trainable A/B₀/P/C/D/O与真实FM完整梯度，保留教学重放及公共参数两侧信用。
继承AdamW、scheduler/clip，原LR在floor1e−5；不得以新sampler重置学习率时钟。
保存分叉来源及新增0/108/216完整ECP；108仅恢复，不物化/闭环/选点。唯一比较终点为新增216（累计2556）。
拓扑按实际吞吐选择并正确承接，不固定原四卡；两臂保留相同逻辑随机流，正常数值差异无需逐tensor追平。

## 4. 固定行为与完整评测

训练侧九项固定54行：3目标×3教学scene×3接收scene×init0/1。
teacher task对应同目标与教学scene，视频由canonical video_schedule中的
`reference_demo_index(20261008, "libero_90", source_local_id, init, demo_count=50, sampling_mode="without_replacement")`确定；
同一teacher条件在三个接收scene复用同一套完整LoRA，不重新挑视频/融合参数。
这是18个教学条件、54次闭环；父T2340、Within216、Product216各做一次，共162行。
init0/1为合法官方non-held source任务初态；若实际资产没有该入口，先报告具体边界，不换成挑选的起点。
官方source任务horizon/场景资产沿现有support评测口径，先登记后运行，不按结果改时限。
本面板是训练侧诊断，可能与训练视频重合，不称训练未见、official400或充分泛化证据。
父模型读回解释既有难度/天花板，三点完整执行；不按小面板分数停训、挑点或自动晋级。

两个唯一终点分别做现行Validation8 correct400、same-task-other400，共1600行。
沿T既有state/video/RNG映射、canonical video_schedule和原50视频无放回合同；同臂跨checkpoint不改变映射，
correct/other逐行视频不同。四面板都是完整单checkpoint，不做80行筛选、结果并集或checkpoint融合。
这两套合法视频是本批预注册qualification读回；其余视频因果controls只在未来模型正式冻结后登记。
复用父T的既有161/150及强MT153原行，不重跑已有面板。新增9项后不能据旧MT153宣称同数据公平突破。

总计1762个新完整episode，独立报告162训练侧与1600官方行。继承render256/model224、双相机180度rotate、
8维state/7维action、10flow、前5执行后replan、settling10、suite horizon、成功即终止与cost-balanced persistent队列。
保存全部raw rows/官方成功、env/policy RNG和生成参数provenance。九格面板各point全部compact，
对front目标的全部9个scene pair/init0共9条/point加full RGB；官方各endpoint correct每task init0共8条加full。
共43 full，其余compact，沿当前capture保存真实动作/完整提案/T+1 state及谓词，不新增控制或标签消费者。
固定审看父/两终点front的SCENE1→3和SCENE3→1/init0六clip，以及两终点task23/39 correct/init0四clip；
每clip八均匀真实已存时点双RGB。其余不声称全量审看；没有terminal RGB时直接注明。

统计逐task/suite、breadth、R/G/L/churn/Jaccard与完整成功集合；父→各终点、Within↔Product、各endpoint correct↔other分别报告。
九格分goal、teacher/query scene和对角/非对角，区分已有成功保持与新得；描述性task簇不确定性不充当独立重复。
两套视频的分数不平均成新qualification，不以某项接触/位移替代完整In/任务成功。

## 5. 事前裁决与后续责任

- Product只比退化的Within好，而完整能力仍弱于原T/强MT：不视为有意义的修复，不自动扩大训练。
- 仅FM、LoRA不变性或九格改善，官方两视频无实际收益：承认局部效应，降低其作为总体解决方案的优先级。
- 两视频中的完整净收益、绝对能力和任务覆盖共同改善：提高该学习关系的优先级；再由main登记同数据充分M/L、
  相邻稳定性或必要重复，以区分视频增量、一般控制学习和偶然交换。本批本身不认证稳定突破。
- 未建立训练侧功能、或所得被能力损失抵消：撤下“改跨情境配对是当前优先修复”的判断；
  不把阴性改叫bug，不自动换encoder/rank/LR/seed后清零累积证据，也不证明所有跨情境学习不可能。
- 临界效应按完整幅度、任务分布与不确定性解释，不由显著性二分替代科学判断。固定终点消除事后挑checkpoint，
  不增设历史145/180等门槛，也不要求逐条保住T成功。

书入caddy的数据不提供task23开柜后退出接触/放碗或task39入炉腔的完整连接，不能预言必然修复这两类瓶颈。
它们在held各占50行，仍是总体性能的重要不足。配对阳性也不自动支持“教学情境已解耦”或“视频胜过语言”。
main收到整批后核原件、解释正反结果并形成下一项完整方法取舍；尚有可做工作且无真实阻塞时继续推进，
不用“本批无自动后继”解除项目责任。停止有限假说不等于停止EMBER，也不等于无限续训保护该假说。

## 6. 资源、工程和交付

新run root：`/data1/user/ymdai/ember_runs/cross_context_pairing_20261008`。旧数据/权重只读复用。
历史两段450更新约7.93/7.75 GPUh、一次400约0.92 GPUh；本批训练约7.5 GPUh，四面板及bank/小面板约5–6，
加工程/profile/失败预计15–18 GPUh；按有效4–6卡与串行实现开销预计6–10wall小时，实际承接起硬12wall小时/20 GPUh。
所有GPU驻留、加载、物化、EGL、smoke、失败与重启全计，不只计训练。实际资源不满足该预计时报告差距。
保守新增峰值上限96GiB，预计bank33–66GiB、ECP/事件/capture不超过20GiB；复用共享A只在现有真实消费者支持时采用，
不为省空间另建格式。launch前在strg01核data1独立user quota与实际用量，并核共享容量/峰值；不足不得切回data0。
每次launch同时live检查两节点，遵守仓库总8/紧张6、单节点6与非干扰共驻规则；NCCL_P2P_DISABLE=1与NUMA保持。
按实际吞吐选择rank、microbatch/frame chunk及独立训练/物化/评测并行，不加无依据的整批4卡硬限制。

实验session负责实现、针对性实际消费者验证、源码/恢复/配对/资源和Git闭环；main不重复工程验收。
实现先验证从T父点恢复与信息墙，再验证query重分配携带同tau/noise、同macro任务/loss权重、完整教学梯度和ECP恢复。
最多额外4次宏更新/448 query用于一次独立smoke及恢复检查，不并入科学训练；最长实际条件最多3次28-query
forward/backward用于物理吞吐比较，0更新。不新增环境smoke行，先完成正式行验证消费者并保留其结果。
科学语义不变的工程修复可在独占分支完成、通过真实消费者检查、push并重新detached冻结后自行继续；原失败和费用保留。
改变模型/标签/梯度/采样权重/选择/评测或超预算，须向main报告具体边界；不将所有工程细节变成审批。
代码从clean pushed commit的detached worktree运行，canonical只留一个active面；历史T由Git和原件保存。
保存run contract、允许列表/事件/分配、完整ECP与manifest、原始行/aggregate、资源账/失败、completion及一份canonical分析。
本批完成后释放GPU并由执行者处理专用入口生命周期，科学证据和当前依赖保留；main消费后决定后继，不能先删有效依赖。
tracked/Git窗口串行交接。只在整批完成或真实科学/预算/有效性边界向main发一次有来源标识的消息；
main active时Steer，idle时Queue。无心跳、分阶段自通知或同一进程已有持续等待时的自Queue。
