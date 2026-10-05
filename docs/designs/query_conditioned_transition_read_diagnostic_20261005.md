# 固定T基础上的转移读取：线性预编译与执行时读取的有界比较

2026-10-05。科学依据为findings§336–338；是否正在执行只看progress。
这是读取方式的机制比较，不是新canonical Writer或完整方法资格。执行时Reader违反单LoRA部署形式，其成绩只能作诊断。

## 1. 选择、事实与可失败预测

固定8条task39教学均可见搬杯前提前转斜腕部，而T2340/C900的100条自身接近多近竖直；强MT50亦如此。
训练36任务并非没有这种控制：task29全部50条教学在首次闭爪命令前最大倾角48.3–82.4°，T1800/C900已有实际大幅转腕；
同类杯的task34/73教学却通常近竖直。它具体化了跨对象/目标组合选择操作准备的缺口，没有证明某抓法机械必要。
另一个已核例是task16：正确物体的固定读取重分配有真实局部作用，却没有新增正确获取/完成。
这两类失败共同反对“只缺一个对象或姿态标签便可补全控制”，也不能由它们断言T不读自身状态。

本批检验：**同一已学教学转移内容，在尚不知道自身状态时预编译成线性映射，是否比保留逐转移内容、
在实际自身hidden上作非线性选择更难学成有益调用。** 后者可能避免把多个教学阶段的作用预先混入同一局部映射。
T的完整深层policy本来是非线性、有反馈的；这不是“静态LoRA无法反馈”的论证，也不是结构不可能定理。
自身与teacher hidden的域差、现有key的布局依赖、Value内容不足和B输出空间限制都可能使新读取仍失败。

选择固定强T2340作共同基础，两个小读出从同一零输出开始，完整36-task普通跨episode FM，比较唯一终点。
这比从新的弱底座开始更直接地测试现有内容的调用机会；代价是结论只适用于该冻结基础及下述有限函数类。
若执行时读取仍无完整行为优势，降低“冻结T特征加另一种逐层读取就能解决迁移”的整类主张，不继续加层、温度、标签或预算保护它。
若有优势，也只说明这种完整读取参数化值得重估，不自动证明线性LoRA不可表达、Compiler是唯一根因或可成功蒸馏。

## 2. 最近似历史与实际区别

- Video Functional及VL版本有联合视频表示/动作Reader，原生中层版本亦已训练，Reader功能风险未胜学生；
  不称此前没有自身query、真FM或联合信用。
- 9/26 native conditional Reader只有两处Q/V注入，完成4步工程smoke，630学习撤回；不是科学阴性。
  当时已提出非线性query读取，不能把概念当本轮新发现或把撤回合同自动恢复。
- 10/2 state-coupled live/stop读取38个**已压缩Z的列**，只在末端加7维速度残差，并同时训练原学生/辅助目标。
  实际Reader控制器19/23低于对应学生20/24，主张“多query信用就更好”已受负证据约束。
- 本批两臂都固定成熟T，复用逐转移的原生K与原Value中间量，在原全部38处作用；新参数、数据、唯一完整FM、
  起始函数及最终B输出空间相同，差别是下述读取算子。没有辅助教师/学生双loss或先强教师再蒸馏的自动课程。
- 旧change-clock增加早期信用却没有相称完整收益，LocalField/NativeCorrection已有真信用及参数消费者。
  新Reader若胜出仍不能把收益单独归于取消擦除、softmax、更多非线性或某帧；本批隔离的是完整读取算子的替换。

## 3. 固定数据、基础与特征

父T2340：
`/data1/user/ymdai/ember_runs/operator_read_write_learning_20260928/continuation2340/T/train/attempts/continuation/checkpoints/macro_00002340`。
原source1000、normalization、tokenizer、assets、probe1729/50×32/tau1、双RGB/stride5含末帧及全部38-target rank128/scale1保持。
父公共A0/B0、P/C/D/O和source均冻结；不更新prefix或任何父参数，不读教师动作/state/reward/pose作为条件。
合法教学只用exact language和同task视频。训练task集合精确为seen_task_scope的原24+12，共36；不是D71池。
全50条既有episode仍合法，query与teacher严格不同episode；不新增或合成轨迹。

对每个target l，复用实际T的原生输入X_l[t,j]及末端H[t,j]，所有50个horizon槽完整保留。
令Hbar为原RMS eps1e-6，k[t,j]=normalize(A0_l X_l[t,j])，固定256维转移量为

```
phi[t,j] = GELU(P_l k[t,j] + C_l Hbar[t,j])
           * D_l (Hbar[t+1,j] - Hbar[t,j]),   t=0..N-2.
```

这正是原TargetWrite的O之前实际Value输入，不是另训姿态/动作编码器。它依赖相邻真实帧与语言上下文，
没有RGB平均、horizon平均、teacher动作、未来真实action latent或私有身份输入。
父完整B_T(V)=B0+M_T(V)仍由原T完整递推产生，不用新J替代父O或重复叠加公共adapter。
共同基础随视频变化，但两个臂的父T参数和生成结果来源相同；冻结基础并不意味着query激活可detach。

每个target新增无bias的J_l∈R^(128×256)，以及Q_l/E_l∈R^(128×128)，共2,490,368个参数。
J全零初始化，Q/E为单位阵；u[t,j]=J_l phi[t,j]，e[t,j]=normalize(E_l k[t,j])。
Q/E使两臂都能在父T的128维读取坐标内学习teacher/self匹配，避免把固定匹配错误仅留给某一臂；
它们不能增加A0的输入行空间，不从此声称解决了teacher/self域差。phi仍是原T冻结量，不随新E重写。
两臂参数形状/初值相同，没有独立温度、额外encoder或静态Value旁路；query尺度可由真实学习的Q改变，不作温度扫描。
phi全零时新增作用为零；这项输入依赖不证明已经学得操作语义或视频必要性。

## 4. 两个实际消费者

用Ekey_t∈R^(128×50)、U_t∈R^(128×50)表示e/u每转移的列；自身该target的输入为h，q=A0_l h、s=Q_l q。

**L：rollout前线性编译。**

```
R_0 = 0
R_(t+1) = R_t + (U_t - R_t Ekey_t) Ekey_t^T / 50
A_L = A0
B_L = B_T(V) (I + R_final Q_l)
y_L = W_source h + B_L A0 h
```

仍只部署一套完整38-target rank128 LoRA；J/Q/E、K、phi、R的运行时读取均在rollout前退出。
上述写入采用原FP32 delta_memory语义，不添加scale、clip或额外独立adapter。

**R：保留逐转移内容，在自身执行时读取。**

```
alpha[t,j | q] = softmax_(all real t,j) (s^T e[t,j])
r(q,V) = sum_(t,j) alpha[t,j | q] u[t,j]
y_R = W_source h + B_T(V) (q + r(q,V))
```

key已单位化，score就是上式点积；无另一个sqrt/温度/自身单位化的隐式约定。每个实际suffix位置各有自己的q。
所有真实转移/50槽参与softmax，只mask物理padding，不按人工阶段选帧；没有反转/打乱或其它教学对照。
原video先独立完整按序读取一次，运行时只读合法预计算Ekey/U；不读新teacher帧、特权数据或环境结果来更新参数。
新增作用在原38处各自投影输出上，继续原o_proj/残差/MLP/后续层及完整十步flow，不另跑一个动作策略。
它虽然使用同一B_T输出空间，但由于r对当前q非线性，一般不能预合并为固定LoRA，因此不属于EMBER部署候选。

J=0时两者都精确表示父T函数；这是实数图的性质，不新增逐bit一致要求。
两臂主FM皆对完整输出求导，须保留r/R、上游h及全部新J/Q/E之间的真实信用；不能将父权重冻结误实现为整条policy no_grad。
两者均无辅助loss、蒸馏、cotangent标签、任务私有参数、RL或task-local优化。

此比较同时改变保留/擦除及非线性归一读取方式，不能再把差异拆成未经干预的独立因果归因。
相同J/Q/E形状和B_T空间控制容量/初值差别，但不保证梯度尺度或优化难度相同；这些属于所比较参数化的实际条件，不做额外归一补丁。
两扩展函数类不互相包含，共同包含的是J=0的父T；R不是所有固定LoRA的已证上界，L也不是全部合法Compiler的代表。

## 5. 固定学习与保存

两臂各270更新，从上述J/Q/E初值与fresh AdamW开始；lr1e-4、betas(.9,.95)、eps1e-8、wd1e-4、统一clip1、无scheduler。
沿既有64步机制学习使用的optimizer口径，只增加到覆盖完整36任务的固定窗口；不作LR/rank/seed/读取scale小扫。
复用T原FormalData/events：seed20260928、逻辑macro1..270、每update四task各28 query、相同teacher/query/frame/noise/tau流，
共30240次query/臂，每task840次。复制的是取样合同，不恢复原T optimizer或声称这是T的exact-resume。
task等权、真实full50×7 FM、offset1、原padding/normalization保持。不得因物理分片、长视频或两臂速度不同改权重。
公开原始宏步/FM统计和逐task风险，不以训练loss选点或删query。不得按本批held结果改变训练/读取。

保存90/180/270的完整新参数、optimizer、sampler/cursor、RNG/拓扑及冻结父来源；前两点只供恢复，不物化/评估/选点。
只读取270一次。协议中J全零的初始函数检查不算新增checkpoint候选。
若数值非finite/OOM或真实接口故障，按已授权工程边界修复/恢复并计费；科学阴性不作bug修复。

## 6. 唯一行为面板与读回

1. 两臂270各做一次完整correct400：validation [3,6,11,16,23,26,31,39]×init0..49。
   原T2340 correct400作为强父参照直接复用。严格沿原scene、teacher无放回映射、env/policy RNG及official协议。
   R使用同一行的合法teacher；运行时保留memory不改变评测环境口径，但结果明确标为额外Reader诊断。
2. 固定训练能力参照仅task29/34/73×原seen144 init32..35、原teacher/scene/RNG。
   父T2340及L/R270各12条，共36条；此前T1800四条只作历史，不替代本父点。
   这是有限训练诊断，四条teacher各一次，不称正式400或held选择面板。

合计**836条新完整闭环**；没有每checkpoint或每模块的额外筛选。每臂validation八task的init0 full，
训练三task的init32三消费者full，共25条full，其余811条compact；全部保存真实action chunks、执行命令及T+1连续状态。
保持render256/model224、双相机rotate180、state8/action7、10 flow、前5 replan、dummy10、suite horizon、成功即止。
原生自身EEF quat/position、object positions、goal predicates均按既有owner记录；不增加接触oracle、环境干预或teacher姿态标签。

报告每task/suite、总分、breadth、全部R/G/L/churn/Jaccard；与历史T比较保留不同执行日期/正常数值波动，
两新臂互比使用同批冻结消费者。没有相邻节点，不能声称已具备正式checkpoint选择资格。
task39保留全部50的杯位移/首次3cm事件/倾角/是否入原生region，task16保留正确及错误物体运动；
训练三task保留提前转腕、目标运动和完整完成，不能只报“腕角像了”或隐藏同动作却失败的病例。
这些描述均不作抓稳、必要姿态或单一失败阶段真值；未存full RGB的行不得用别例画面补过程。

在同一真实forward被动保存38处新作用相对父LoRA输出的聚合RMS、R读出的有效帧/槽分布摘要及非finite/shape检查，
按task/condition/flow保留；不存全层全部巨大attention张量、不追加探针或把熵/幅度当成功标准。
同一行的原始预测/行为才是主要证据；训练末记录/400行为/资源和负例齐全后统一裁决。

## 7. 结果分支和停止线

- R比L和强父T取得有覆盖的真实新增控制，并保留多数原能力：提高“按自身情境保留并读取转移内容”的支持。
  单task、单姿态或参照退化造成的差额不够；即使满足，也不直接启动蒸馏或声称最终LoRA已有效。
- L同样改善或更好：收益不能归给在线读取；完整合法线性编译仍有当前数据内的学习机会，但这项冻结基础诊断不是fresh方法资格。
- R/L近似、只改善内部FM/角度、或以广泛丢能力换局部收益：不继续放大这个冻结特征读取族；
  保留真实局部正例，降低它作为主修复的优先级，不以预算未穷尽延长、加头、温度或换key扫描。
- 两者在有监督训练任务也未取得相应控制：明确该构造尚未提供有能力的读取函数，不能把失败归为LoRA压缩。

所有分支均在本批终点停止、整批回main。没有自动fresh、另一Reader、更多closed-loop、controls/Test或RL。
此窗口不能解决整个EMBER目标，主讨论须结合原件形成后续判断，而非把完成诊断当最终成果。

## 8. 工程、资源、并行和生命周期

唯一root：`/data1/user/ymdai/ember_runs/query_conditioned_transition_read_20261005/`。
预计含工程4–6小时，硬wall8小时、**12完整GPUh、24GiB新增峰值**，含加载、profile、失败、冻结树、cache/临时文件和输出。
依据旧S64约14336 query/.427 GPUh，本批60480 query约1.8 GPUh的原路径基线；逐转移Reader及native重读预计使学习/物化到3–5GPUh，
836条完整行为约2–4GPUh，另留工程profile/失败余量。这是预算估计，实际早期吞吐须更新ETA，不能冒称已测。
预计新增约20GiB：L的412套bank约9GiB、轨迹与紧凑读回约4–5GiB、恢复/代码约1GiB、受限临时cache≤4GiB，另留失败余量。
R的全部400逐帧native不永久物化，按condition流式复用；不得默默建立几十/上百GiB全池缓存。
若实际profile预计无法完成固定范围，及时报告科学/预算边界，不静默减少面板、精度或合同。

启动前由实验session核strg01 data1独立quota/个人实占/共享容量，live检查两个GPU节点和实际当前占用。
遵守现行全局8卡/空闲≤10时6卡、单节点≤6；没有本批额外收紧的总卡数上限，也不因两臂各world_size而限制整批并行。
训练/物化/评测按真实吞吐并行，已就绪L物化/评测不等R；多rank不跨节点拼片，NCCL/NUMA/拓扑合同保持。
有显存余量须实测更大microbatch/frame chunk/适当persistent workers，用吞吐及峰值决定，不只沿保守默认。
profile只用本批已注册训练输入，最多每臂两次可丢弃更新后恢复J/Q/E、optimizer/RNG；无额外环境smoke或科学case。

复用canonical T native/TargetWrite、LoRA安装、FormalData、FM、scene恢复/视频调度和官方evaluator；
新增入口/模块为本诊断所有者，不复制长期trainer/evaluator，不修改旧冻结树或恢复退休路线。
工程从main独占codex分支/worktree，应用结构检查，验证后合并push并从clean pushed detached frozen运行。
main派发后交出tracked/Git写窗口，不并发写；工程修复/检查/Git与运行闭环由实验session完成，不等主讨论重复工程验收。

必要消费者检查为：J=0父函数、L真实预合并、R全38真实读取、仅J/Q/E参数梯度且上游激活信用保留、信息墙/有效batch/配对、
完整恢复及受限storage。按实际风险验收，不引入逐tensor/逐bit一致、hash sidecar或新通用测试框架。
完整物化只在270，R不能误走纯bank消费者；L部署不得残留读取hook，原38-target正式消费者保持可用。
完成后退役本批专用入口/hooks，保留Git/frozen、学习checkpoint、来源、全部原始行为/汇总/失败/费用及completion。
只直接等待进程退出事件；不固定轮询、不分阶段Queue或自通知。整批完成/真实边界后，仅一条带来源回报main并交回写窗口。
