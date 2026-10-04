# 教学角色到实际 Q 方向的编译：固定父点的有界学习辨识

状态：固定64、197/960及独立科学消费均已结束，见机制§133/findings§312与progress顶部。
下文保留原冻结科学合同，不再授权新增运行、扫描或自动后继。

2026-10-04，机制分析§132。实际接手、版本和运行状态只看progress。本项是分析阶段的三臂学习，
不是完整Writer的formal fresh或C900续训，不扩充数据；没有自动400、续训或下一架构资格。

## 1. 问题、干预和匹配参照

task16原八格只有W40/scene46/noise46正确搬butter，其余搬orange或ketchup；同C900公共四格也没有取得butter。
已有自身attention移植能转移部分对象获取，却不能独立转移完整能力；坐标辅助更准未胜普通FM，原生方向直接搬用在换位后多数反向。
因此检验的不是“已有正确公共能力被视频破坏”，也不是“看对位置就足够”，而是：
**把教学中指定角色的原生视觉内容与自身真实目标/干扰key差共同用于学习Q参数，能否得到可迁移的角色条件控制。**

F为同预算普通FM；G只给现有完整生成LoRA增加自身角色信用；R在G上增加§3的实际角色编译及其教学选择监督。
G区分单纯的自身角色信用，R对G区分新增的整条角色绑定路径及其必要教学监督；后者不是纯参数量或单一loss的孤立消融。
三臂均开放原38处A/B生成头共同适应，避免将冻结下游的一次改向当成完整学习上限。新路径不被当作已确认的唯一修复。
强T161/MT153的整体资格及已有task16强参照继续保留，本诊断不能以只胜C140代替最终目标。

## 2. 父点、固定数据和学习口径

父点：`/data1/user/ymdai/ember_runs/conditional_read_write_continuation900_20261002/conditional_read_write/train/attempts/continuation/checkpoints/macro_00000900`。
C900训练/物化85919994、450父a0e0248d、source1000训练b8ea00e9保持；本次运行身份另列。
source、公共A0/B0、native计算参数、四层c/d解释器冻结；F/G/R均更新所有ConditionalTarget头。
R的新增模块fresh初始化，三臂fresh AdamW：lr1e-4、betas(.9,.95)、eps1e-8、weight_decay1e-4、整体clip1、无scheduler、seed7。
仅64个有效更新，父引用和64完整恢复点保留，不读取中间点作行为选型。

只用原36-task allowlist内以下8项；coverage_v1的24/8/8及既存数据规模不变：

| global task | 正确被操作角色 | 原任务 |
| --- | --- | --- |
| 12 | salad_dressing_1 | salad dressing入篮 |
| 13 | bbq_sauce_1 | BBQ sauce入篮 |
| 14 | ketchup_1 | ketchup入篮 |
| 15 | tomato_sauce_1 | tomato sauce入篮 |
| 17 | milk_1 | milk入篮 |
| 19 | orange_juice_1 | orange juice入篮 |
| 43 / source3 | butter_2 | 后方butter入上抽屉并关抽屉 |
| 96 / source56 | butter_1 | butter入tray |

43的butter_2由实际BDDL的init/goal确认，不能误用前方butter_1；完整Close目标仍保留。
六种篮中角色与两种合法butter任务提供当前已有的组合支持，不增加任务/示范/外部来源，也不声称它们独立覆盖所有角色规律。
各task只用既有teacher demo0/1，共16条件；选择在本批结果前固定。query demo2..29为28条跨episode主监督，30..49为本次不回传的B20。
它们是本次头部适配的分离查询，不冒称C900从未看过这些episode。

更新step=0..63，block=step//2；用`default_rng(SeedSequence([7,132,block]))`排列sorted八task，
偶数step取前四、奇数取后四。每选中task的两teacher共享28 query；每步8条件、每条件权重1/8，query各1/28。
各task32次、每teacher32次，共14,336 FM condition-query/臂；三臂43,008次。
query帧f用`default_rng(SeedSequence([7,132,task,step,demo,0])).integers(0,episode_length-1)`，
flow seed用同tuple末项1的`SeedSequence.generate_state(1,dtype=uint64)[0]`取低63位，
交原canonical FM/noise/tau消费者；两teacher及三臂复用同一query/noise/tau。
B20的每episode固定一帧，以`[7,132,task,demo,2]`同法取f，官方十步noise seed以末项3生成。
实际完整query manifest须在学习前保存；额外角色forward复用FM的epsilon，不推进另一条随机流。

## 3. 唯一新增R路径：角色方向与动态使用系数合成为Q因子

保留canonical完整N×50的X/H、四层因果c/d与原S/M；frame stride5及真实末帧规则不变。
在同次合法teacher native中额外读取18层、两真实相机512个图像位置的**RoPE前prefix K**，每key256维、KV头数1。
这些K来自真实冻结PaliGemma prefix，仍含SigLIP位置和前层上下文，不是位置无关语义。
teacher只有RGB/exact language，无State；不能拿其position_ids/RoPE代替自身完整prompt的真实位置。

记rms0为现有`x / sqrt(mean_channel(x^2)+1e-6)`，无bias、零输入仍为零；t=1..N-1是到达帧，j=0..49。
以到达c[t,j]选择origin图像K[t-1]，所有图像位置均可读，无mask/对象名/pose条件输入：

```
alpha[t,j,p] = softmax_p( (Pq rms0(c[t,j]))^T (Pk rms0(K_0[t-1,p])) / sqrt(256) )
p_l[t,j] = sum_p alpha[t,j,p] rms0(K_l[t-1,p])
u_l[t,j] = P_l p_l[t,j]                                  # 256
r_l,h[t,j] = D_l,h rms0(d[t,j])                           # 128
R_l,h = sum_(t,j) u_l[t,j] r_l,h[t,j]^T / ((N-1)*50)       # 256×128
R_l = concatenate_Q_heads(R_l,0 ... R_l,7)                # 2048×128
A_Q,l = A_C,l ; B_Q,l = B_C,l + R_l
```

18层各自P_l:256→256、八个D_l,h:1024→128；共享Pq:1024→256、Pk:256→256，均无bias。
P_l初始化identity、D全零、Pq/Pk用torch Linear默认初始化；全部新初始化在seed7独立fork_rng内，不能改变公共query RNG。
共20,381,696新增参数，R按原Compiler口径FP32累计。D零使初始R严格为零，原完整策略保持；第一步D与teacher选择项可有梯度，P_l的功能信用从非零D后取得。
不要求identity初始化每个上游首步梯度非零，不通过增scale、换初始化或延长更新追阳性。

这里只汇聚已经构成的角色×动态算子，未平均原始frames、H、跨video features或最终LoRA。
c/d先完整保序计算；没有额外rank8/memory bank、位置搬移规则、对象字典或另一套部署adapter。
d=0时R=0，语言/静态选择不能单独产生本新增Value；这不等于实际动态已被证明具有操作语义。
其余20个target保持原完整A_C/B_C。最终仍只有38-target/76因子、rank128/scale1：`(B_C+R)A_C`，不增加部署rank。
Writer一次编译后退出；自身hidden经同一A、Q、真实RoPE/自身K、Value、o_proj/MLP、十步flow影响动作。
第一层h主要由noise/time决定，后层才有自身视觉反馈；不能把每层的使用系数先验称为对象状态。

source/公共/解释器不更新，故X/H/c/d/原生K可以按父点只缓存一次；A/S、最终A关联key/delta-z/Value/M、alpha及R每更新重算。
不沿旧固定A缓存合同复用依赖A的量，不复制大source/原始数据，不落盘所有query的巨大prefix缓存。

## 4. 标签只用于loss，信用经过实际参数消费者

训练8task使用既有HDF5 states、实名BDDL/XML及可见visual geoms，复用VisibleObject已验证的CPU segmentation/时刻合同。
obs[f]对应states[f+1]及一个2ms运动学回退；不环境step、不生成新示范。按实际原RGB相机分辨率恢复，
沿原180度rotate、224和16×16 patch映射生成两相机可见像素质量；不标不可见轮廓、不改crop或手画训练ROI。
teacher只标上述正确被操作实体及后代；query同时标全部可见可搬物体，排除robot、fixtures和basket/tray等目标承载容器。
43保留butter_1作干扰，不能把两块butter合并成一类。执行者核实际body/geom registry与少量固定首帧叠图后沿同一规则生成。
标签无效/不可见不丢输入，loss该frame/query置零，记录全部有效数，平均分母仍为登记frame/query数。
不读task16或其它held教学state/action/pose作标签；任何标签、body表、路径/ID均不进入native/解释器/Compiler条件。

**教学项（仅R）**：与K同一origin帧t-1的正确实体可见像素在512 patch上归一得qT，`aT[t]=mean_j alpha[t,j]`。
不得把到达帧t的mask监督到origin K[t-1]。
`L_T = mean_t KL(qT||aT)/KL(qT||uniform512)`；只在目标可见且分母非零时计算。
chance为1，真实选择alpha直接产生上节p_l，不是独立辅助预测头；不同j仍可有不同选择。

**自身项（G/R）**：同一个query的真实obs/state/exact language及原epsilon，以tau=1做一次实际完整LoRA forward。
这是额外一次有梯度消费者，避免用含真实action的FM x_tau直接承担角色读取；不增加新样本/动作标签/监督伪动作。
记其第l层、head h、实际前5槽i对图像p的真实logit为`ell_lhi,p=(R_i q_lhi)^T Ktilde_l,p/sqrt(256)`。
用可见实体o的面积归一patch分布q_o构成

```
s_o = mean_(18 layers,8 heads,first5 slots) log sum_p q_o[p] exp(ell_lhi,p)
rho = softmax_over_visible_movable_objects(s)
L_Q = mean_query [-log(rho_correct)/log(number_of_visible_objects)]
```

目标不可见或可见候选不足2个时该query为零。实体面积归一避免只因大物体像素多获胜；不是强迫全部图像attention落在物体上。
L_Q约束目标相对其它可搬物体的读取密度，不保证图像总质量或每个head都正确，不能以它单独证明角色控制。
所有18层/8head/前5槽预先固定；不能事后选层、头、ROI、温度或改对象表。
实际Q/h不能detach；自身余切包含经h返回前层Q/V/action-in及最终A/B的联系，和FM一起回传全部原生成头、R新增模块。
冻结source仍保留对LoRA/hidden的导数。真实prefix K、position_ids、mask/GQA沿当前query，不能用teacher R或手写位置代替。

每condition主项为原unmasked完整50×7 FM，query均权；固定
`F: L_FM`，`G: L_FM + .1 L_Q`，`R: L_FM + .1 L_Q + .1 L_T`。
L_T每condition每update只计一次，不因28 query或physical rank重复；所有condition仍按1/8。
总G/R新增28,672次tau1 condition-query消费者，全部计费用；主FM的tau、future约定、归一化及动作offset不改变。
不做坐标回归、专家蒸馏、teacher动作辅助、公共FM、RL或一致性loss。

## 5. 固定终点读回与197条新闭环

三臂只物化64；每condition只生成一套完整LoRA，跨自身init/layout/noise复用，不逐轨迹适配。

1. train8×teacher0/1×init32/33，每臂32；父C900在相同16条件也读32，共128新增行。
   复用`operator_seen_task_diagnosis_20260929/attempts/scene_canonical144/scenes/`，这16个物理scene已核存在。
   这是有限池训练侧面板，不冒称50-video无放回；同task两teacher在相同自身state/noise下配对。
2. 每新臂读取task16原scene0/46×teacher47/40×noise0/46的八格，以及原init0..7×原/XY交换布局的16格。
   后者teacher顺序固定47/33/28/46/32/1/24/43；完全承接`object_position_transport_20261004`的物理变换。
   两面板的scene0/teacher47/noise0/original重合，只执行一次并引用；原scene/env/policy种子、56项时钟和condition已核同源。
   因此23不同case/新臂，三臂69行。原C900八格/原换位行只读复用，所有已有数值分叉均并列保留，不挑高值。

总197新增环境行。train组每task/每模型固定teacher0/init32为full，共32 full/96 compact；task16全部69 full。
合计101 full/96 compact。完整双RGB（full）、全部50×7 chunk/实际physical actions、state8、T+1所有body/EEF/quat/gripper/官方goal保留。
task16仅butter官方In触发成功，orange In被动记录；其它task保留全部原生goal（包括43 Close）。
官方source/normalization、256→224/rotate、10flow/前5、dummy10后scene恢复、各suite horizon和成功即停保持。
root7、canonical CPU50×32及绝对replan noise合同不变；teacher与physical init/noise身份分列，不伪造formal video ordinal。
强T/MT旧task16原/换位面板只读引用，其缺少的交叉情境不补造分数、不追加强参照运行。

另在固定B20每臂16×20=320 query执行一次官方十步动作读回，共960，0梯度；不另做A28矩阵。
记录真实同次forward全部层/头的前5槽角色分数、图像总attention质量、各物体密度及动作前5/全50误差，
动作同时保留有效future与unmasked、motion/gripper及单通道。B20标签仅来自训练8task，本次不进入更新。
R另读其实际`R_l A_l h`对同次Q图像logit的贡献及去掉该直接项时的局部密度差；只作同h/key下的算子读回，
不另forward/环境，不称去掉R的完整策略或因果中介占比。原A/B和下游共同适应不能由此拆干净。
物化时保存真实alpha的逐帧50槽平均、R_l与完整因子；不据此挑teacher。训练实际loss/有效标签数及完整更新日志保留。

## 6. 裁决及继承停止线

行为是主要证据：按task/suite/teacher/scene/layout/noise报告官方成功、正确/错误实体的完整过程、R/G/L/churn及集合重合。
body中心、3cm抬高、EEF接近或夹爪命令不冒充接触/抓稳；full RGB必须核正反例。多模型/teacher复用的物理初态不当独立样本。

- G相对F有可迁移的正确获取及完整收益，而R没有额外改善：支持现有结构可利用这种角色信用，降低新增编译的必要性。
- R在B20获得目标/干扰关系，同时相对F/G在多个原任务/teacher和task16原及换位情境改善正确操作与完整成功、保持已有正例：
  支持这条角色绑定路径的有限实证，不只凭单一旧成功格或总分抵消；仍不证明动态必要性、完整架构泛化或超过强MT。
- 读取分数/teacher选择变好而控制无相应改善：降低角色选择及本编译足以修复调用的假设，不再追加同类lambda/层/rank/seed小扫。
- F同等或更好：保留普通FM获取，不把其收益归给新监督/结构。若只适配已见组合、task16不迁移，明确区分获取和组合迁移。
- 若64步尚未获得相关读取关系，只认定本次固定特征/有限学习未给出修复依据，不说所有视觉角色不可学，也不自动扩容量或续训求通过。

全部分支到此停止。原生K未被假设位置不变，图像读取未被当作全部控制；上述两项历史阴性不能被一次局部阳性清零。
后续科学选择由main核实际原件后作出，不把本暖父点局部分析换名为完整fresh方法资格。

## 7. 工程、资源与交付

唯一root：`/data1/user/ymdai/ember_runs/native_role_binding_compilation_20261004/`。
预计含标签/工程/并行三臂/全部读回/退役**4–6小时**，硬限**12完整GPUh、40GiB新增峰**，估计峰28–34GiB。
依据旧两臂各14,336 query＋96闭环为1.338完整GPUh；本次三臂、两臂额外tau1、197行/960十步query和新标签/原生K增加成本，
预计完整计算约3–6GPUh，工程/标签约2–3小时，准备工作可并行但科学采样与权重不变。10GPUh处按实际剩余预估是否能在硬限完成。
显著超期或预算无法满足须报告具体原因/剩余，不静默删行、降低capture、排除失败或扩大计算。

启动大cache/学习/环境前核strg01 data1独立quota、相关实占/峰值和shared；每launch前live同时查gpu01/02、当前/启动后卡数。
沿仓库8/6合计、单节点6及真实A40吞吐安排，三臂和已就绪物化/读取可并行，不另设由单臂world size推来的整批上限。
全部新写data1，大模型/数据/父checkpoint/scene只读复用；不缓存所有query的prefix，不为填显存新增病例或profile矩阵。
每臂最多两次可丢弃profile更新，随后恢复父/新模块初值、optimizer与RNG才计64；profile计完整预算。
R第二次实际更新可核上游功能信用，聚焦真实标签对齐、Q/RoPE消费者、完整余切/冻结边界、缓存依赖与单LoRA，不新增逐tensor/hash审计。

既有实验session独占工程/canonical tracked/Git，复用canonical native/FM/target replay/evaluator，按结构风险使用code-architecture-gate。
新模块按本诊断职责分离，不复制policy/evaluator或留下另一条canonical训练入口；主讨论不重复工程审查/测试。
实际消费者检查后push、clean detached运行；已定位工程违约在原合同内修复/new push/freeze并保留失败，科学改变/预算越界交main。
保存完整恢复（含optimizer/RNG/cursor/topology/schema）、标签/query provenance、训练/新读取代码身份、banks、raw/aggregate、completion及全费用/退出。
整批后退役临时入口/hooks、封闭新执行，源码由Git/frozen保留；集成push、清理task-owned工程树后交回窗口。
执行者一次持续等待实际子进程退出，不轮询或selfQueue；整批完成/真实边界只发一条有来源回报，已接手/完成不重复启动。
