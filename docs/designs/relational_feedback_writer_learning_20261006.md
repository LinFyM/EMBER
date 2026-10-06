# 关系监督、训练用反馈函数与完整 LoRA：首批共同学习

2026-10-06。Owner明确当前目标是超过强MT，评估教学侧仍只有视频和既定exact language，
允许充分使用既有训练LIBERO数据，随后授权“接下来你自主推进，尝试下这条路”。
这撤销此前等待讨论的暂停。本文件由progress登记后成为唯一active design；不恢复已结束RL或旧局部诊断。
未来换视角/跨本体是适用方向，不是本批实现或准入要求。

## 1. 要改变的学习联系与既有反证

具体不足是教学展示的操作关系没有稳定变成自身状态下的完整控制：task39的准备与后续放入、task23开柜后的取放均有缺口；
但train29已有目标相关准备，task23有条件成功，类似手角也可能一成一败。不能归因为没有姿态信息、完全没有反馈或缺一种固定阶段。
T2340=161/400、MT300=153/400保持强参照；新去噪回报72=158、63=154，seen SDE109→109，并未提供新的收益。

本批检验：用真实训练物理关系建立低维、跨episode的控制学习任务，能否让RGB条件生成器获得更有用的函数监督。
改变的是一套完整学习机制，不声称已隔离每个组件的贡献，也不以其它路线失败证明它正确。
旧双域实际A坐标降低误差但未修复控制；Video Functional联合读出/蒸馏未学成强教师；条件速度C(V)Uh共同FM仍退化。
因此本批既不能只加pose head，也不能把另一个RGB读出称作可靠teacher。新F使用实际物理关系与当前query状态，
其自身能力和最终单LoRA能力分别读出；任何一个局部阳性都不是下阶段自动资格。

## 2. 唯一部署图

```text
exact language + 原双RGB教学（stride5，保留真实末帧）
 → 同一fresh公共beta下的真实native读取H[N,50,1024]与38处X
 → 预测手/场景实体的有序几何与语义轨迹（不接收真值）
 → 共享关系计算与全视频上下文
 → 各层条件A/B编译：A=A0+S，B=B0+M
 → 一套完整38-target、rank128、scale1 LoRA
 → 原冻结source读取自身RGB/state，官方10步ODE、执行前5动作后replan
```

source和图文prefix冻结；native仍使用合法完整RGB/L、probe1729、tau1和50-horizon，不新增假图像/动作输入。
公共A0/B0、视频读取、关系处理和Compiler全部fresh共同学习；不继承T/C或RL权重，不预训MT再冻结。
K=1，不声称dynamic-K。Writer只运行一次，闭环没有F、物理标签、对象表、运行时视频读取或第二adapter。

## 3. 物理标签和信息墙

固定原36-task allowlist、每task50 episodes、source-only normalization；精确来源沿
`configs/operator_read_write_v1/continuation2340_spec.json`的source和events字段，事件只取首450步。
不新增task、示范、人类数据、自身纠错采集或RL。validation/test的教师state/action/pose不读取、不缓存、不产生梯度。

训练标签恢复沿Git `7bd8a6f5:scripts/audit_operation_semantics.py` 的已验证合同：逐episode model_file，
`states[i+1]`按实际MuJoCo timestep回退一个积分子步再forward，对齐obs/RGB[i]；不能用默认场景替代episode XML。
EEF的p来自obs ee_pos，R由obs axis-angle ee_ori转换；官方p取grip site、R取eef body，不能混用site xmat。
末帧没有下一full state，保留RGB并仅mask缺失几何标签。query本来不采最后一帧，action offset1不变。
工程须在所有36任务各一个实际episode的首/中/末有效点核对同步；未知不静默补值或删task。

实体表只在标签/F侧：公开BDDL对象和非table/floor fixture的root body，加有内部slide/hinge的可见moving body；
同一个body只保留一次，所有有内部关节的drawer/door都保留，不以柜体根部代替抽屉。机器人自身只用一个hand。
固定一个hand槽和32个匿名实体槽；在建全标签前审计36个场景registry的数量，超过32则先回报具体schema边界，不能截断。
采用编译模型body/site的实际坐标，不把body原点叫视觉中心、抓取点，不平均mesh顶点。
保存p、R、双指原始qpos、内部slide/hinge qpos及各字段mask；不构造grasp、成功阶段、contact正确性或人工子目标标签。
双指qpos不冒称指垫净宽。内部关节类型/地址从episode模型解析，禁止跨task硬编码qpos下标。

语义标签为公开对象类别及moving body通用部件名，去掉实例数字、场景名和路径；不含task ID、OOI、goal或程序。
用冻结source词嵌入对名称token求均值，按固定seed20261006的正交随机投影得到64维unit向量；投影不学习。
这是标签/F字段，部署预测连续向量，不把GT实体名称或实体数量传给Writer。
teacher/query的GT实体对应仅由同task原registry建立，F无task embedding、实例位置embedding或task-ID route。

manifest现有元数据为1,800 episodes、284,735 raw frames、59,114教学帧；57,314可配几何标签、1,800末帧mask。
只为这些教学帧及预注册query帧恢复标签；不生成全量RGB/模型/特征副本。离线姿态规模预计小于1GiB，实际以registry为准。

## 4. RGB关系表示与完整编译算子

**视觉读取Phi。** 对token RMS后的真实H，1+32个共享256维slot query做两层cross-attention/FFN，8heads、FFN1024；
随后每slot沿真实时间做两层双向attention/FFN，同width/heads。位置只进Q/K，采用real frame_index/5的RoPE；dropout0。
不平均frames或50个H token。全部层独立参数、pre-LN。hand槽固定；匿名槽用整条video的一次Hungarian匹配监督，
代价在有效帧平均位置、旋转、语义误差，不按每帧重新交换身份；匹配是loss索引，不成为部署条件。

每槽预测p3、R6、语义64、存在概率、none/slide/hinge概率及两类q；hand另预测双指q。
R6由两个3向量Gram-Schmidt形成旋转，数值eps1e-6。位置单位米，slide按.25m、hinge按pi、双指按.04m缩放。
预测槽的mask/存在权重也来自模型；GT数量与mask不得teacher-force到真实G调用。未观察/对称旋转可能不可唯一恢复，不能把低误差当完美物理感知。

**关系处理Omega。** 只消费上述预测字段，不能把Phi的无约束hidden绕过物理字段作为动态Value。
各frame用两层共享关系attention处理节点：
`r_ij=R_i^T(p_j-p_i)/.25`，`Q_ij=R_i^T R_j`，加各自局部gravity、语义、关节/双指字段；
edge的256维MLP既进入attention bias，也进入Value，节点存在概率同时作用于权重和值。
再用每实体两层双向时间attention/FFN取得w[t,i,256]；width256、8heads、FFN1024、dropout0、位置只进Q/K。
除hand类型外不加实体索引embedding，因此后段对匿名槽的联合置换等变、最终读取不变。

对每个native horizon位置，以H[t,h]的256维投影查询w[t,:]，得到context及同attention下的`w[t]-w[t-1]`；
分别投影成c[t,h,1024]与bias-free d[t,h,1024]，d首帧为0。完整视频上下文可以决定早期准备，
但没有把teacher时刻当自身执行时刻。沿`ConditionalTarget`现有FP32 delta_memory公式，
用真实X[t-1]、c[t]、原H[t]和d[t]共同产生S；再用最终A0+S重算keys并产生M。38处均保持，A/B都能条件化。
不恢复旧CausalInterpreter的平行动态流，不把几何只放进被丢弃的辅助头。

这仍是学习式函数编译。部署修正为`B(V) A(V) h_own`，38层与后续非线性共同作用；A决定响应哪些自身hidden，B决定改什么。
共享F的任意非线性不保证可以精确编译，也不由这种表达式推出已经获得对象/阶段语义，须看实际单LoRA控制。

## 5. 独立训练用反馈函数F

F与G不共享可训练参数，GT轨迹只进入F；绝不能通过F调用向Phi/Omega/Compiler灌入GT条件。
它接收A示范的GT关系轨迹、B的当前GT物理状态及真实自身source FM query，A/B严格跨episode。
query的RGB/state、noisy action z和tau沿原FM合同；从未挂载LoRA的冻结source同次forward取得H0[50,1024]与v0[50,32]。
teacher actions/terminal/reward不进F，F的动作监督仅是B的原训练actions；GT teacher末帧按已登记mask忽略。

F的独立关系encoder采用§4同样的两层空间+两层时间256结构，读取GT而非预测字段。
当前B使用同一F空间encoder；每个teacher时间/entity形成Value：teacher节点、当前同实体节点、
`r_hi(A,t)-r_hi(B)`、`R_hi(B)^T R_hi(A,t)`及关节/夹爪差，由共享MLP投影256。
全程teacher上下文保留对象间关系；当前物体位置与关节变化可改变读取，无教师时钟、人工阶段ID或已完成目标丢弃。
以投影H0的完整50 query作两层cross-attention/FFN，width256、8heads、FFN1024，最后预测7维residual。
`vF[:7]=v0[:7]+residual`，其它25维保留v0；输出层零初始化，所有其它层正常fresh初始化。
此结构把“相对当前状态的关系误差→动作修正”作为共享输入，不假定线性伺服足以完成接触/多目标任务。

F是待学习模型，不是oracle。已有真实状态/action能够定义第一轮训练，不要求先获得新的纠错数据。
它没有新增信息；潜在收益来自较直接的状态—关系函数及有限样本下的归纳偏置。仅在专家访问分布训练，纠错覆盖仍是限制。

## 6. 梯度、损失与固定学习窗口

同原事件流seed20260928，450 macros，每macro4 tasks×28跨episode queries=112；每task50次访问，全部50 teacher各一次。
一次更新完成全部四task后再step，逻辑query/noise/tau、任务1/4权重和完整50×7均值不因物理卡数改变。
训练样本的同一个z/tau/y供实际LoRA FM、F FM和蒸馏；F不读取query未来状态或action以外的额外未来标签。

`L_G = L_FM(vG,y) + .1 L_relation(Phi(V),qA) + lambda(u) L_KD(vG,stopgrad(vF))`
`L_F = L_FM(vF,y)`，`lambda(u)=.25*min(u/90,1)`，u为本次更新1…450。
G完整图始终接受真实FM和蒸馏；relation只经视频预测图回传，包括同一native公共beta；F只接受L_F。
不存在先训F再冻结、逐模块课程、public FM、RL、trust回滚、错视频训练或学习式动态loss权重。

L_relation为六个实际存在字段组的等权均值：p/.25 SmoothL1、旋转chordal误差、1-cos语义、
平衡正负presence BCE、关节类型CE及有效q误差、hand双指误差；各组先按其有效实体/帧均值，不能让slot数或视频长度改变task权重。
旋转用`.25*||Rhat-R||_F^2`；joint/hand q采用上述固定尺度的SmoothL1，joint类型与q在组内等权。
匿名空槽只贡献presence，未知物理字段mask不伪造0标签；辅助一次/condition，不能随28个query重复加权。

两个fresh AdamW分别更新G、F：lr3e-4、betas(.9,.95)、eps1e-8、wd1e-4、各自clip1；
沿原150-update warmup、1200-update衰减与floor1e-5的scheduler公式，不在450终点重新定义LR时标。
G先按原identity seed20260721初始化A0/B0，模块seed7；F在独立seed20261006 RNG作用域初始化，不能扰动G/RNG事件。
normal BF16/TF32、FP32 optimizer/存储及必要累积。每90宏步保存完整G/F、两个optimizer/scheduler、sampler/cursor、rank RNG、topology/schema。
物理rank1…6由实测吞吐决定，允许完整checkpoint边界迁移并记录；不把4 logical conditions变成全批4卡上限。

平方误差下，蒸馏把实际输出目标改为`(y+lambda*vF)/(1+lambda)`；它会引入F的偏差。
只有关系状态对动作足够、F学得更好等条件成立时，才可能降低有限样本学习难度；无限数据下没有凭空增加真值。
不能从tower property或非零VJP宣称受益。报告F、G真实FM及完整闭环，保留F比学生弱、teacher好而学生坏两类反例。

## 7. 完整读出与裁决

唯一候选固定450；360只作相邻学习证据，不取二者最大值。两点均用原official ODE strict correct400，
validation IDs[3,6,11,16,23,26,31,39]、states0…49、K1全50视频无放回、video schedule seed7和原T固定映射。
执行规格完全沿AGENTS§8；复用T161、MT153的原400，不重跑、不降低强参照。

450另做原36-task seen144：states32…35、seed20260928、与最近父T seen144相同teacher/scene/env/policy噪声。
G为合法单LoRA；F在同144仅作明确标注的训练任务privileged诊断，使用实时GT状态和训练teacher几何，禁止用于held/Test。
F仍官方10步ODE/50×32/前5动作/同horizon，不执行梯度或改checkpoint。父T ODE144复用132+原12的完整已登记来源。
合计新1088完整episodes：G360/450各400，G450 seen144，F450 seen144。不能只交内部误差或F成绩。

每一臂保留per-task/suite、breadth、retained/gained/lost、churn、Jaccard与全部raw rows；G两点对T/MT及彼此配对。
主要目标为有覆盖、相对稳定地超过强MT；T161继续显示收益/代价，不设置逐例保留T的额外硬门槛。
单点刚超过MT不冒称稳定突破。只有完整G有实际收益才讨论后继必要视频增量；本批不自动开controls/Test、续训或参数小扫。
F及G seen都弱：降低本次关系控制学习假说支持；F强而G弱：定位到视觉获取/编译联系仍未传递，不能称LoRA容量不够；
seen变好held未变：没有获得所需迁移；G改善但关系/F证据不支持：只登记经验收益，不补写唯一机制故事。
450仍学习时如实报告，固定窗口不是整个架构已被证明不可能；任何后继都回main结合完整证据决定。

记录全部真实continuous动作/提案与原生谓词；validation每task init0、seen每task init32保留full双RGB，其余compact。
选择性可视检查应列实际clip与已捕获时刻，不补拍终态、不把移动/接触叫稳定抓取。
已有task23/39缺口及train29/42/73的关系/准备/衔接用完整行为解释，不为得到局部阳性增加人工阶段指标。

## 8. 工程、资源与交付

根`/data1/user/ymdai/ember_runs/relation_grounded_writer_20261006`。全部新增产物只在data1，历史资产只读复用。
预计含标签/工程/训练/物化/评测8–12小时，依据最近完整Writer450窗口及新增F/source query和标签恢复成本；
硬16 wall-hours、40完整GPU-hours、96GiB新增峰值。预计训练18–28GPUh、读出4–6GPUh，其余供实现/profile/失败；这是预估，须用真实profile更新。
标签CPU历史6668点/84.34s外推教学约12分钟，连同query、各场景编译和工程按1–2小时预留，不把外推当实测。
接受时记起点；达到12h或30GPUh且剩余无法在硬预算内完成，回报精确已完成范围/缺口，不偷偷删面板或加预算。

实验session独占实现/Git窗口：独立codex分支/工作树、比例适当的实际消费者检查、集成main并push、新clean detached冻结再formal。
使用formal-training-launch的一份实际命令/依赖/输出/拓扑/恢复合同；GPU launch前查双节点与strg01独立quota、个人实际用量和共享容量。
总卡数遵AGENTS的8/6与单节点6；跨阶段独立工作按可用吞吐并行，不跨节点拼训练碎片、不dummy占卡。
profile最多3次丢弃宏步更新，只用首批合法训练事件，之后复位G/F/optimizer/scheduler/RNG；验证更大的microbatch/frame chunk，
保留samples/s、LoRA/s、最长视频峰值和停止放大的实际理由。验证梯度/信息墙/完整checkpoint，不做逐tensor低位一致或新增hash。

新增owner按标签、关系模型、训练F、实际功能信用编排内聚分工，复用native读取/完整A-B算子/原事件和官方evaluator。
不要继续把新科学逻辑塞入已大型的operator_writer/run.py，也不要另造评测器或第二套FM/data runtime。
F消费者为有allowlist保护的训练诊断，绝不能成为部署fallback；当前仅保留这一个新科学执行面。
旧实现/负证据由Git与frozen保留；本批关闭时退役无后继用途的专用入口/hooks，保留原始formal结果与完整恢复资产。
已定位且不改变科学语义的工程问题由实验session按AGENTS自主修复/核验/推送/新冻结接续，完整记录失败费用；科学变化/未知或预算越界回main。
main消费科学原件并承担取舍，不重复工程验收。整批完成或真正边界才一次回报main并交回tracked/Git窗口；
长任务只等实际退出/checkpoint调度事件，无轮询日志/心跳/逐面板自通知。可靠唤醒沿已核实的官方Queue/resume通路。
