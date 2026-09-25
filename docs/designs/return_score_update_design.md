# 去除执行无关幅度信用后，一次更新是否真正改善闭环

2026-09-25。是否active只看progress；机器合同`configs/return_score_update_v1/experiment_spec.json`。
主讨论01a0cd94-65da-7b22-8ca9-7ba35f454632；具体执行仍为Sol 01a0cd90-ebb7-77a1-a20b-a858825d2f66。
固定两个独立单步候选，第一阶段只有96条配对闭环；不自动接续训练或补评。

## 1. 从已验证计算机制到未验证的行为作用

§148确认：只替换五维夹爪幅度score的条件期望，完整Writer方向相对改变量.587716、cosine .837657，
约旋转33.1度；RAW重算对E2仅.001609相对差，不足以解释该变化。
但偶/奇梯度cosine仍−.02686/−.02450，task37份额从.68625升至.71362，没有解决信用集中与两半不一致。
因此拒绝“只是不重要的数值变化”，也不接受“信用已经可靠或性能会改善”。
该计算机制只针对本次Gaussian回报估计，不解释此前监督/视频不足的全部根因。
§147中正反方向均26/64、父35/64，§145的方向代理失效、§146的单点夹爪代理效应、
旧private Writer正证据及缩头/全任务隔离阴性继续约束解释；不以某个梯度模块norm大就冻结它。

条件期望理论保持理想score的期望并减少一种条件方差。若J_Sigma局部L-smooth，则
`E[J_Sigma(phi+alpha*g)] >= J_Sigma(phi)+alpha*||grad J_Sigma||²−L*alpha²*E||g||²/2`
提供动机；它不是单批收益保证，也不把探索目标J_Sigma等同于canonical无探索J0。
本批直接检验：以同一个SGD系数实施这项修正，是否减少旧更新造成的能力损失并改善实际成功？

## 2. 唯一学习干预与固定来源

父仍为C_S00@1155（7dc95edb训练），从完整保存权重读入；Source1000与normalization冻结。
只读复用`/data0/user/ymdai/ember_runs/return_score_conditioning_20260925/gradient/`
的`RAW_total.safetensors`和`RB_total.safetensors`，来源d6348660399d0a24841af30ed9a98ff64b5078e3。
不重估梯度、补采集、读FM/expert标签或混入旧Adam。两个候选各自从父创建一次fresh SGD，无momentum/decay/clip。

令原已固定r=.17076466164261486，`alpha=r/||g_RAW||`；分母只用已封存d634的完整RAW梯度范数，
不得根据本批成功或动作变化再校准。三格为

- P：未更新父；
- RAW：`phi_P + alpha*g_RAW`；
- RB：`phi_P + alpha*g_RB`。

两更新的**SGD系数完全相同**；RB不再独立归一到r。已知RB步长约为RAW的1.0546倍，
这属于替换梯度、保持优化器的有限作用，不把它伪称等参数/函数步长比较。
选择同一系数是为了只更换信用估计器；不另加尺度控制或按结果改变alpha。
保存完整Writer权重、实际位移/系数/参数名映射、fresh SGD step1状态与父来源。
它们是两个独立一步候选，不称父Adam exact resume，不作为新选定模型或继续训练的起点。
支持直接复用现有candidate_step等所有者，以各自`alpha*norm(g)`作为实现半径即可；不另造trainer。

## 3. 最小独立闭环面板

保持原八个合法训练任务`[2,5,12,17,22,25,34,37]`，不按前轮涨跌或梯度大小筛任务。
每task固定states32、33与正确teacher46、47交叉，三臂每格32条，共96条。
取旧面板自然顺序前两个state，未使用新分数选择；34、35本批不执行。
这是受已知旧结果启发的同任务机制补验，不称全新盲测、50-video无放回、strict400或部署资格。
两个teacher均仅RGB/exact language输入Writer，actions/state/reward/ID等仍不进入部署条件。
每臂每task每teacher编译一次完整38-target rank16 LoRA，共48个bank，两个state只读复用各自bank。
P、RAW、RB均在本批同一clean pushed detached实现下物化与评测，不把旧E2的35/26或其subset拼入主对照。
这可同时控制正常重编译/闭环变化；旧R不是新RAW，历史self差异单列。

canonical协议不变：双相机render256/model224及rotate、8D state/7D action、10flow、执行前5后replan、dummy10、
成功即停、原suite horizons、env/policy seed7；无探索、在线Writer调用或task-local适应。
三臂同task/state/teacher的环境与policy噪声配对，两teacher也共用该state的RNG。
复用canonical persistent evaluator与动态队列，不新增平行rollout引擎；物理卡分配不得改变面板定义。

正式pilot为task2/state32/teacher46三臂各一条，共3条，包含在96内；检验实际生成/加载、成对首态/RNG、
轨迹/被动trace/谓词/成功终止与退出后，运行其余93条。pilot验收不看谁分数高，不增加工程episode或环境探针。
工程CPU候选/元数据及prepare验证在formal前完成；真实环境接口以这3条为准，失败保留原件并回报，不自动重跑。

## 4. 读出与事前裁决

预定比较只有RB−RAW、RB−P、RAW−P；每teacher单列与两teacher等权总计。
报告八task/四suite、breadth、R/G/L/churn/Jaccard，所有父成功保持与丢失，不只看净分。
20000次joint bootstrap/seed20260925：先重采八task，再在每task内重采两个state，两teacher始终保持配对；
三臂及全部对比复用抽样，95%区间取共同样本的2.5%/97.5%分位数。小面板/单采集与训练seed，
只能作描述性区间，不能靠某个对比显著选模型。

全部96条保留compact动作/噪声/state、T+1 object/EEF/gripper及stage predicates。
full cases固定task2/12/22/34、state32、teacher46、三臂，共12条双相机，覆盖四suite。
利用本批轨迹已有的首轮full50×7输出和前5实际动作，配对报告RAW−P、RB−P、RB−RAW的
xyz/rotation/gripper函数变化及开合sign改变；不新增冻结query/10-flow预测。
首轮共96个输出已经属于闭环，不能重复统计为额外独立样本；后续状态分叉不冒称同query差额。

预定解释：
- H_noise_harm：RB相对RAW改善两teacher实际成功和父能力保持，才支持去掉该分量有行为价值。
  若仅恢复RAW损害却仍低于P，只称部分减轻损失；不能称解决EMBER、视频有益或已有可采纳改进。
- H_no_benefit：参数方向已实质变化，但RB没有相对RAW的成功/保持改善，或改善一项伴随另一项大损失；
  本有限更新不支持沿该修正继续训练，不以更小步长/更多训练/换任务挽救。
- H_remaining：P仍优于两候选，或不同task/video交换明显；仍有其它信用噪声、有限步/分布迁移问题，
  不把未测解释当已证实原因，也不自动扩大回报采集。
- H_function：仅首态参数代理明显、执行指令变化很小，最多限制此早期接口；不由首态一处外推整条轨迹无作用。

若样本不足以区分值得保留的竞争解释，主讨论可另行说明追加信息价值；本合同**不自动授权**其余state、
held400、更多teacher/seed、负方向、FM格或连续RL。不能从96条挑最佳节点，两个候选同时完整报告。

## 5. 执行与资源

study `/data0/user/ymdai/ember_runs/return_score_update_causality_20260925`。
新data0≤3GiB，开发+正式代码≤768MiB；复用父/旧梯度/Source/数据，旧树与原件只读。
预计.8–1.2 GPU-hours，硬上限1.5（含CUDA初始化/物化/工程/失败/评测），最多同时两卡、项目总计≤6。
候选构造可CPU完成；实际GPU资源由Sol按两节点live身份/所有权/余量调度，开新root前核对strg01独立quota及共享余量。
从最新main隔离实现、验证、集成push，再冻结一个新提交用于全部新候选/48bank/96闭环。
只对当前phase声明的scope提供入口，不放宽旧研究bank/selector校验或热改旧formal树。
新CLI应薄且复用既有生成/被动采集/评测机制，不扩成通用实验框架。

保存精确launch命令/环境/版本/预算、48bank与两候选provenance、96原始行、12full索引、连续trace、
三个比较/成功集合/首轮动作读数、退出回执与completion。计时覆盖完整GPU进程，不用reserve冒充实测上界。
同一冻结提交可只补未开始的登记面板，不重复已完成行；实际失败episode/部分控制步必须如实单列，
有工程/身份/信息墙/配对/预算问题即保留证据回报主讨论，不自动重跑或启动下个研究。
正常长任务持续等退出事件，不轮询日志/cache。完整96结束后主动Queue主讨论原件、缺项、资源及完成信号，停止新增实验。
