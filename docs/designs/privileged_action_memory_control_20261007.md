# 给定真实状态对应后，教学动作值能否形成闭环控制

2026-10-07。Owner持续自主授权；由progress登记后为唯一active分析批次。
这是原36个训练任务上的特权控制诊断，不是EMBER候选、合法部署策略或新的Writer训练。
§373停止的物理输入／下游编译修补保持停止；本批不恢复G/F、P/Q或RL。

## 1. 具体缺口、历史与判断对象

G450 seen84/144、F45066/144，准确teacher字段的同图学习P81→78、Q82→81；
task29多条轨迹搬动酒瓶而没有OnRack，task38仍缺第二个壶的On，不能把运动或准确关系等同于可调用控制。
F并非已强、只待压缩的教师。另一方面，原跨episode FM确有真实动作监督，不能说整个系统没有反馈学习机会。

已有§81绝对几何1-NN在跨episode完整5×7动作上的MSE为.12443553，低于video_mean的.25459689，
且24/24任务均改善；相对几何.12634790没有优于绝对几何，原相对替换假说已关闭。
这保留了状态匹配传递部分动作价值的正证据，但当时没有source forward或自身rollout，旋转分项也未胜过均值。
本批沿用原**绝对**匹配规则，不另选坐标、邻居数或距离尺度；补的是实际闭环和匹配动作Value对照。

要裁决的是：给定特权状态对应，成功教学中已有的真实动作值，能否比同帧冻结source预测值形成更有覆盖的反馈控制？
若真实动作值明显更好，才支持进一步研究如何在合法训练侧学得这种控制内容及其调用；
若连此固定动作记忆都不能控制，不能继续把成功示范或准确轨迹默认成强功能教师，再投入同一路编译蒸馏。
这不是对F失败的唯一归因；F的参数化、训练与执行图不同，不能由新两臂差额反推其唯一根因。

## 2. 唯一主变量与实际算子

```text
同一条原教学episode → 真实几何key q_i ──────────────┐
  demo_action：原记录actions[i+1:i+6] → value_i      │
  source_action：同帧双RGB/L/8state → 冻结source     │
                 → 原10步ODE完整50动作 → 前5值     │
                                                   ↓
每次自身replan：真实q_own → 原绝对距离1-NN → 取一个value_i → 执行前5动作
```

两臂拥有相同teacher、key集合、匹配算子、物理单位、初始场景与环境随机流，唯一设计变量是记忆动作值来源。
自身轨迹随动作而分叉是该干预的结果，不强制后续查询或检索索引相同。
没有学习参数、FM/RL、梯度、专家拟合、LoRA生成、LoRA平均、第二adapter或checkpoint选择。
teacher动作、teacher/own真实几何与BDDL对象身份在这里都是明确特权输入；两臂均不能计入EMBER成绩。
只有既定exact language进入source预测；BDDL身份只建立同task点对应，不伪称RGB理解或合法task泛化。

令e、R_e、g为官方obs的EEF位置、axis-angle对应旋转矩阵、双指qpos。
对象来自该task官方BDDL的`:obj_of_interest`；每个对象包含名字等于obj或以`obj_`开头的全部body/site，
按(kind,name)排序。p_ok、R_ok为这些点的实际位置与旋转，先在同对象点内平均，再对对象等权，记mean_ok。

```text
D_common = ||R_e−R_e'||²_F/2 + mean_ok(||R_ok−R_ok'||²_F/2)
           + ||g−g'||²/(2×0.04²)
D_abs = D_common + (||e−e'||² + mean_ok(||p_ok−p_ok'||²))/0.10²
i* = argmin_i D_abs(q_own,q_i)
u_own[0:5] = value_i*
```

平局取最早teacher位置；每次replan可重新选择任意合法位置。没有时钟/单调阶段mask、阈值门、平滑、
插值、速度外推、相对坐标替换、剩余轨迹平均或source自身动作残差。不能从结果再挑邻居/teacher/动作seed。
此算子只提供沿示范邻域的分段常值响应；它不保证轨迹外恢复，也不包含完整接触、动力学或阶段状态。
索引倒退/停滞和距离只作行为描述，不直接命名为错误阶段或根因。

## 3. 固定数据、几何与动作缓存

唯一allowlist为`configs/operator_read_write_v1/seen_task_scope.json`的原train24＋non-held support12：
`[0,1,2,4,5,7,12,13,14,15,17,19,20,21,22,25,28,29,32,34,35,36,37,38,42,43,51,55,56,62,64,73,95,96,97,101]`。
每task使用原seen144的init32…35、seed20260928及原state-video映射，每condition一条teacher；
不换scene、另抽video、扩任务或读held/Test。固定每task四个条件，不能称整轮50视频无放回或strict paired400。
复用原50教学池、HDF5、normalization、canonical source/assets/manifest；只为144条实际需要的teacher恢复几何。

teacher合法位置固定`i=0,5,...`且`i+5<N`，两臂共同使用这个mask。
真实动作Value严格为原HDF `actions[i+1:i+6]`，继承post-action offset1；不平均或重标动作。
这会舍弃没有完整五个未来动作的尾部采样位置，须记录完整/使用indices；不能冒称与旧完整视频G输入相同。
几何恢复直接沿用Git `9f90a14d:scripts/audit_cross_init_relation_retrieval.py`的`geometry_episode`定义：
原XML和`states[i+1]`，按`mj_integratePos(...,-model.opt.timestep)`回退传感器子步，再`mj_forward`。
使用canonical资产映射，离线EEF同步保留原1e-4m要求及实际误差；全部点的signature必须与同task自身环境一致。
不将body/site集合缩成根中心、静态region或新的Phi槽；旧关系cache只有实际signature和时刻一致才可复用。

source_action使用原Source1000：`configs/operator_read_write_v1/learning_spec.json`的source checkpoint、
processor/tokenizer与冻结source normalization。输入为同一teacher位置的原双RGB、exact language、记录的8state，
正常原生prefix和完整10步ODE，固定公共Gaussian初始噪声seed1729、shape50×32，每帧同一个噪声样本。
不使用fake/zero image、隐去原生prefix、真实action作去噪输入、LoRA或任何task expert。
保存实际完整50×7反归一化输出，以前5作为memory Value；所有source权重冻结、eval/inference_mode。
source只做一次离线Value缓存，随后释放；两臂自身rollout均不再加载或调用source模型。

live查询直接取实际sim的同名body/site p/R及官方缓存EEF/gripper obs；不rewind或额外step自身环境。
记录live缓存与sim派生EEF的被动差异，保留finite/schema；不要把离线同步阈值套到live传感器缓存而改变输入。
真实Value是原7维控制量，source Value是原processor反归一化量，统一使用canonical动作执行约定；不另加servo或缩放。
若通用capture接口要求50个提案，demo臂从同位置保留可用真实未来动作，末端复制仅用于未执行的padding，
明确valid length/mask且前5必须全是真实值；不能把padding称为真实完整预测。实际控制与科学比较始终只消费前5。

## 4. 唯一闭环面板与证据

两臂各144，共**288条新episode**，固定一次完成；没有更多seed、80/400、额外训练/held/Test或controls。
复用canonical scene、dummy settling10、success即停、suite horizon220/280/300/520、前5动作后replan。
双RGB采集及source图像处理遵原render256/model224与旋转合同；teacher存储图片按原processor处理，不虚称其原生尺寸为256。
本控制器是privileged NN，元数据显式标注；**10 flow steps仅属于source Value物化，不能说NN闭环本身执行了10步flow**。
两臂严格配对task/state/scene/teacher ordinal/env RNG；自身actor无随机采样，policy RNG沿原映射保留但注明未消费。

主比较demo_action对source_action，报告全部per-task/suite、target24/support12、breadth、R/G/L、churn与success-set Jaccard。
旧T2340 seen109（原132＋12）、MT93、G45084、F45066只引用原行作背景，不重跑；
它们与特权NN的输入和执行方式不同，不能把跨这些方法的净差称为单变量因果效应或超过MT的EMBER成绩。
另对同一memory第一5动作报告source相对真实值的归一化MSE及平移/旋转/夹爪分项，沿冻结source尺度。
这是原缓存的一次描述，不增query面板，不以离线误差或检索距离替代闭环。

保留全部实际continuous动作、提案/valid mask、原生目标谓词、scene/teacher/RNG、所选索引/距离和Value来源。
每臂每task init32 full双RGB，其余compact，共72full/216compact；不为补终态重render已结束的轨迹。
固定审看tasks12/25/29/32/38/73、init32、两臂，共12clips×8个已记录时刻×双相机=192张图，
结合全部任务原生谓词说明接近、搬动、目标放置与失败；没有真实接触证据时不宣称已抓稳、碰撞或接触阶段。

- 若真实动作有明显且跨任务的闭环优势，支持“给定对应后，示范控制Value有新增价值”；仍未解决RGB获取、
  learned feedback泛化、一次38-LoRA编译或held迁移，不自动启动F/G训练。
- 若两臂都强，说明该状态对应和source动作先验已有可用组合，不把效果全部归于teacher真实动作或当前Writer。
- 若两臂都弱，或仅个别任务得失，离线动作传递尚不能成为强闭环教师；不接阶段/平滑/残差、K/尺度/seed变体保护此路线。
- 若source更好，保留source先验与真实动作跨状态不适用的解释，不挑任务隐去逆例。

所有分支都按效应大小、原target缺口、任务分布和四初态的不确定性裁决；不把局部阳性拼成完整方法资格。

## 5. 资源、工程与交接

唯一root `/data1/user/ymdai/ember_runs/privileged_action_memory_control_20261007`，新增全在data1。
预计2–4wall-hours；硬6wall-hours、3完整GPU-hours、24GiB新增峰，从实验session实际承接起计，
含工程、加载、失败、profile、缓存、EGL/render、评测、冻结树与临时文件。4wallh或2.25GPUh时，
若剩余预计无法在硬限完成则报告具体边界，不减面板或暗增预算。旧192episode几何恢复99.37秒仅是CPU估计依据，
source推理/scene/render仍有不确定性；估计缓存、72full/raw和工程峰12–16GiB，保留24GiB硬界，无optimizer checkpoint。

承接后独占tracked/Git；先查strg01独立data1 quota、个人用量、shared空间，再创建实质性产物。
每次GPU启动live核两节点，遵总8/空闲≤10时6、单节点6；所有实际source/EGL占用统一计费。
NN actor用CPU，source物化与独立就绪读出可并行；复用cost-balanced long-first persistent队列，
不要为NN加载无用VLA或dummy占卡，不把某个进程world size变成整批上限。
source有显存余量时最多两次首个既定teacher chunk的物理batch/profile，保持同噪声与消费者语义、纳入费用，
保留首个有效cache并按真实吞吐/峰值决定batch；不能拿BF16低位差异当科学故障。首合法条件的运行检查计入288，不加额外episode。

独占codex worktree实现、针对性实际消费者检查、main集成/push、clean detached新冻结；旧树与原件不热改。
复用既有环境/scene/队列/capture，只加有固定allowlist的typed diagnostic controller与最小缓存接口；
不造fake LoRA bank、不复制第二套评测器、不在大run.py堆科学逻辑。几何恢复及距离复用原定义，避免平行版本。
必要检查针对时间offset/点signature/合法五动作/source冻结与输入/配对/恢复，禁止额外hash、全树扫描或逐tensor低位审计。
专用入口与临时注册由本实验session持有，整批结束退役；保留Git/frozen/cache/raw/失败/完成与费用记录。

工程接口违约可在原范围修复、新push/freeze并复用有效已完成行；科学输入/控制规则/面板/预算改变须回main裁决。
只在整批完成或真实边界时一次可靠回报main并交回canonical tracked/Git；不逐面板Queue、自通知或心跳。
main收到后直接核原件、作完整方法取舍并继续Owner目标；此合同不自动授权后继Writer/F、更多诊断、held/Test或部署。
