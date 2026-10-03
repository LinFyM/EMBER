# 实际读取坐标的双域关系监督：固定教学特征的有界辨识

2026-10-04，依据机制分析§124。实际接手、版本、执行与结果只看progress；登记不等于已取得结果。
本项没有选择新的canonical架构或正式fresh训练，不扩大已有数据规模。

## 1. 要改变的失败预测

当前C900的普通B学习在固定条件13→24/32，完整迁移却只有140→143/400；task16仍经常搬orange而不动butter。
同一LoRA在对象换位后既有跟随区域、也有跨位置搬错实体及第三对象的反例，第一次规划已经响应布局。
这些事实没有唯一定位Reader、A或B，却排除了把问题只说成没有自身输入响应或普遍缺少搬运能力。

本次检验一个限定假说：在固定现有教学特征下，让生成A在教学和自身执行两端读出**各自当前的同一角色几何量**，
能降低学习可调用控制关系的难度，并使完整策略更可靠地操作该对象。不是把两个不同状态的坐标强行拉成同值。
竞争解释是这些特征/读写头尚不能获得这种坐标，或坐标可以学准却未形成有用控制，或普通FM已经足以取得相同改善。
它是分析阶段的有界学习，不把几何标签、同A、非零梯度或拟合改善当作根因已经证实。

最近历史边界：旧operation_semantics§2已提出self实际A监督，未实施；本项不得称首次想到语义rank。
VisibleObject已监督teacher实际读取，state-coupled信用已监督真实自身hidden，均有正负边界；J联合位移没有胜过普通动作。
新增的具体联系是对**最终同一A的teacher写入坐标和self实际投影**赋予一致物理含义；原生K归一化、B自由度和其余125坐标仍在。
目标body原点不是抓取点、接触、阶段或控制指令；主FM继续负责实际完整行为，不能套用几何伺服保证。

## 2. 父点、样本与唯一两臂

- 父点为`/data1/user/ymdai/ember_runs/conditional_read_write_continuation900_20261002/conditional_read_write/train/attempts/continuation/checkpoints/macro_00000900`。
  900训练/原correct400来源85919994，450父a0e0248d，Source1000为b8ea00e9；原32行bank读取923ff89b。
  使用它是因为上述固定失败/有限获取原件均在该模型；不以其140替代强T161/MT153作为整体目标。
- source、公共A0/B0、native和四层解释器全部冻结；两臂均更新38处ConditionalTarget的全部A、B生成头。
  不更新公共policy或另加可部署模块。相同完整(A0+S,B0+M)仍是唯一38-target/rank128 LoRA。
- 固定原四train task及teacher：0:(40,11)、12:(25,14)、20:(38,42)、32:(17,43)。
  精确复用`operator_learning_limit_diagnosis_20260930/query_manifest.json`的64步A28、frame、episode及flow seed。
  两teacher共享各task当步28查询并排除自身两teacher；8条件各1/8，每臂14,336次condition-query使用。
- F臂：原完整50×7动作FM。G臂：相同FM加§4的双域坐标监督；除该loss外，其余学习条件完全一致。
  两臂均fresh AdamW，lr1e-4、betas(.9,.95)、eps1e-8、weight_decay1e-4、整体clip1，无scheduler、seed7。
  固定64个有效更新，不以中间结果选点、改权重或续训。保存父引用和64完整恢复点。

原A64只训练B，本批两臂都开放A/B；故它是相关历史，不能与G的差额冒充此次辅助项的匹配效应。
可共用同版本固定X/H/c/d；**A、S、K、delta-z、Value和M每次更新重算**，不得沿旧固定A缓存合同跨更新复用。
优先只读复用实际匹配的缓存/资产，不复制source、原始数据或模型。

## 3. 唯一几何量与信息墙

每个时点的标签为`r=(p_object-p_EEF)/(1 metre)`，三个世界坐标分量；不作均值/方差归一化，不从分数调尺度。
EEF使用与当前obs对齐的原`obs/ee_states[..., :3]`；对象点如下，沿已验证实名schema：

| task | 对象点 | 含义范围 |
|---|---|---|
| 0 | body `akita_black_bowl_1_main` | 正确黑碗原点 |
| 12 | body `salad_dressing_1_main` | salad dressing原点 |
| 20 | site `wooden_cabinet_1_middle_region` | 中抽屉上的实名点 |
| 32 | body `moka_pot_1_main` | 摩卡壶原点，不包含炉钮阶段标签 |

这不是通用active-object或完整任务状态定义。task32开炉阶段也使用同一壶点，原FM和其余坐标仍承担完整任务；能力丢失须保留。
query点位直接复用`joint_action_effect_credit_20261003/labels/query_positions.pt`及manifest；EEF从原合法query字段读取。
teacher只对上述8条既有视频的真实origin帧恢复点位和EEF，沿7bd8a6f5的post-action时刻合同：
`obs[f]`对应`states[f+1]`及已核实的一个2ms运动学缓存回退。仅CPU运动学，无环境step/render/新示范。
旧operation_semantics缓存没有EEF数组、也不覆盖本批全部8条teacher，不能声称已经完整可用或替换为相近episode。
时间/实体对应不能成立时报告具体科学边界，不换点、平均对象或改帧追结果。

训练teacher pose/state只用于**生成之后的loss标签**；native、解释器、Compiler条件始终只读原RGB/exact language。
不得把r、EEF、对象表、state、episode ID或标签mask送入Writer/条件表示。query标签也不回灌Compiler。
不读取官方validation/test teacher的pose/action/reward作为标签；task16的环境自身状态仅作§5被动读回，0梯度。
全部数据来自原36-task允许范围内的既有数据；没有新任务、episode、在线训练状态或外部数据。

## 4. 实际投影和梯度消费者

固定`P=[I_3,0]`，选action_out最终A的第0、1、2行；这是预先固定的坐标约定，不按当前权重或效果挑rank。
其它125行不加坐标标签，仍与其它A/B头一起接受完整FM学习；不改rank、K归一化、递推、source norm、动作normalization或官方flow。
令`A=A0+S`，teacher raw X来自公共β、真实双RGB/无State prefix、固定probe/tau1的action_out投影前hidden；
self h来自装入该condition完整LoRA后的真实FM forward，同样是最后AdaRMSNorm之后、action_out之前的50×1024。
二者条件不同，不能用teacher hidden代替self，也不能把50槽当50个未来物理状态。

```
zT[t,j] = A X_teacher[t,j],       t为native X[:-1]对应的真实origin帧
zQ[q,j] = A h_self(q,j,x_tau,tau)
L_T = mean_origin,50,3 (P zT-rT)^2
L_Q = mean_query,50,3 (P zQ-rQ)^2
L_G = L_full50x7 + (L_T+L_Q)/2
```

每项均乘原condition权重1/8，只计算一次。物理当前r在同一观测的50槽上相同，**没有平均或替换实际输入hidden/frames**。
L_Q与主FM复用同一次真实query forward/noise/time，不增加teacher-action输入、端点或前5加权。
self h必须是当前functional LoRA叶参数的实际forward；`A h`的直接A导数与经h到前面37处参数的导数均保留。
两者连同主FM组成完整76-factor余切，再经同版本Compiler回传到全部A/B头；不能detach h或只训练独立读出头。
L_T使用同次replay的最终A及固定raw X，直接提交A信用；主FM对M(A)的依赖不删除或重复计数。
冻结公共/native/解释器参数不因辅助项获得更新，已有缓存仅限§2允许的固定量。

若写`A h=[r_hat; eta]`，最终速度含真实项`(B0_g+M_g) r_hat`及其余125坐标、source项。
这给坐标一个实际消费者；它仍可能被忽略、抵消，或给出错误作用，且teacher K还除以完整z的范数。
不能由坐标误差低宣称M已经正确调用它，更不能把该层FM速度分量直接称为物理末端位移或完整十步因果效应。
普通FM可能无需此辅助便学得更好，本批F是必须保持的匹配对照。

## 5. 一次固定终点读回

每臂64后仅进行以下两组，共48行/臂、96新增行：

1. 原四train task×两teacher×init0..3，共32行。复用原Original32的scene/teacher/RNG；保持能力与全部得失。
2. 原task16/init0..7及八teacher，按`object_position_transport_20261004`的**同一原/XY交换布局**各8行，共16行。
   固定各自butter/orange的z/quat/qvel、其余scene、robot/controller/time，0额外settling/step；不新选位置或对象。
   新臂每个condition只生成一次完整LoRA用于两布局；不根据环境状态重新编译。

task16父C900/强T2340/强MT300的48行和原train Original32直接复用，保留已知旧/新原布局的数值行为分叉。
不重复强参照、不用反事实布局形成官方分数或选checkpoint。task16只有官方butter In终止，orange In继续被动记录；其余任务仍按各自官方目标终止。
每臂8 full：原六full（0/12/20第一teacher init0，32/t17 init2、t43 init2/3），加task16/init0原/交换两行。
其余40 compact；总16 full/80 compact。完整goal/continuous/所有对象/EEF/gripper、真实actions和50×7 chunks保留。
Source、原scene、root7/绝对policy noise、双256→224/rotate、state8、10flow/前5和suite horizon全保持。

两个新臂均被动保存每replan实际10步flow中全部50槽的`P A h`，连同观测步索引；不增加forward。
环境当前的同一实名点与EEF只用于退出后的几何误差读回；task16点固定为`butter_1_main`，不传入policy。
通过每condition既有完整A/B银行读取B0/M的相应列，可解释该层直接分量；不追加rank消融或因果中介主张。

另在原固定B20读取两臂官方十步动作、同次P A h及当前query几何，保持原共同噪声、160条件-query/臂。
报告动作前5/全50、有效future与原unmasked口径、motion/gripper及各通道；teacher坐标以原8条固定X直接读回。
报告坐标在真实部署10个tau/前5与全50的误差，不以含真实动作的FM拟合代替部署坐标质量。
训练日志取实际已计算loss，记录几何信用确实进入完整因子；无额外A28矩阵、拟合探针或parent模型重算。

### 5.1 首次规划tau=1的角色变化读回（2026-10-04训练前补清）

不新增样本、标签来源、loss、forward、环境、模型或预算。复用本节已经被动保存的真实`P A h`。
对task16原/交换全部init0..7与F/G，取first replan的第一个tau=1调用，保存所有50槽：
`Delta_rhat[j]=(P A h_swapped)[j]-(P A h_original)[j]`。
相同condition/LoRA、机器人state、language及初始Gaussian保持；比较发生在任何新动作之前。
从两行真实initial continuous的butter、orange及EEF world位置构造三维`Delta_r_butter`、`Delta_r_orange`，
固定世界点模板为`-Delta_p_EEF`（本物理合同机器人不变时应为0）。不以理想XY反向/零值替代实际数组。
所有8 init/两臂/50槽全部保存；对三模板分别报告RMS误差，first5和full50并列，无拟合、阈值或选例。
不能以更接近某模板认定内部唯一原因；与两臂完整闭环、独立B20/部署几何共同裁决，没有自动晋级/扫描。

## 6. 什么结果会改变判断

主要行为比较为G对F：四train任务能力、task16原/换位各自官方成功、完整对象操作、R/G/L/churn及不利个例。
八物理初态在多模型/布局中重复，不冒称96独立样本；不把中心靠近、3cm或gripper命令当成抓稳。

- 若两端坐标在学习/独立B20及实际部署有明确获得，且G相对F在多个条件/原与换位布局更可靠地操作正确对象、完整能力得益，
  提高这项实际读取约束的支持。仍未分别隔离teacher项与self项、静态/动态视频来源，也不证明整体超过强MT。
- 若坐标变准但控制未改善、只使某个中间量漂亮，降低“这种角色坐标约束足以改善调用”的支持；不追加λ、rank、层或标签扫描。
- 若F已取得同等/更好改善，保留普通FM获取，并降低该辅助作为主要修复的优先级。
- 若固定特征/64步下坐标未获得，只认定这次有限候选没有提供依据，不说所有几何不可学；不自动扩大网络或延长以求通过。
- 若只在原四任务有效、task16无相应迁移，不把已见拟合拼成可迁移方法；完整报告条件范围和能力交换。

所有分支均到64及上述读回为止。没有official400、相邻选择、正式fresh、其它布局/对象、controls/Test、RL或数据扩展。
阳性只是对同一困难问题的有限机制依据；没有自动晋级合同。整批后main依据原件独立裁决。

## 7. 执行、预算与交付

唯一root：`/data1/user/ymdai/ember_runs/role_coordinate_credit_20261004/`。
预计含工程/标签/两臂学习/读回/退役**2–4小时**，硬限**6完整GPUh、20GiB新增峰值**；预计峰12–16GiB。
依据原两臂固定B64约.81GPUh加环境.22GPUh；本次开放A图、增加实际坐标余切与96行/新视频编译，给予明确余量。
工程约60–100分钟、标签约5–20分钟、并行学习/读回约40–90分钟；实际profile修订预期，不静默超预算。
5GPUh时若剩余不能在6内结束，报告具体剩余与成本，不删不利case。全加载/native/profile/失败/IO/退出均计费。

实验session独占工程与canonical/Git窗口，负责现有Compiler/FM/evaluator内的窄接入、针对性消费者核验、push与clean detached运行。
主讨论不重复工程测试。检查覆盖真实A/h余切、冻结边界、标签时刻/实体/信息墙、相同query/noise、缓存依赖及原换位物理语义。
普通BF16/TF32差异可接受，不新增hash、逐tensor精度审计或全仓测试。所需8 teacher几何恢复不是扩大示范数量。
启动前live双节点、当前/启动后卡数、data1独立quota/实际用量和shared容量；所有新写入data1。
两臂、编译和读取按真实吞吐并行，cap沿仓库规则，不另固定低于它的整批卡数上限。
有余量须在既定输入内验证更大的microbatch/frame chunk/合适分片；最多每臂两次可丢弃完整更新，恢复初值/RNG后才计64。
不为填显存增加query、重复病例或profile矩阵；完整50与有效权重保持。

正常长任务由执行者一次持续等待退出，不周期读日志、不发阶段自Queue。
整批保存合同、实际标签/代码身份、完整恢复、LoRA banks、原始动作/几何/行为、全部失败、汇总和费用/资源退出证据。
结束退役专用入口/hooks，Git/frozen/科学原件保留，push并交回窗口；只发一次整批结果或真正科学/预算边界。
接收方按root判重，已开始/已完成不得重复执行；整批交付后停止新增计算，由main接续科学判断。
