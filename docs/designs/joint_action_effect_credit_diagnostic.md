# 动作与物体位移联合信用：固定教学表示的有界学习辨识

2026-10-03。依据机制分析§116登记；实际接手、版本、运行和结果只看progress。
本项检验已有教学表示到自身控制的学习接口，不是新的canonical Writer或完整fresh训练。

## 1. 需要辨识的实际不足

当前条件读写900在原task32两teacher×四初态仅1/8成功。init2的交叉续行表明，两套完整LoRA
从prefix17到达的状态都能完成，从prefix43到达的状态都未在原时限内完成；它们在切换时都已开炉。
因此单说没有第二阶段知识或没有目标predicate并不充分。它没有把差别唯一定位为抓持、接触或几何误差。
旧T2340的S/P/D64又说明普通FM本来就能学出一些搬壶修正，也存在开炉、放置及跨teacher的交换。

本次假设：在已有教学表示和读取地址下，动作标签以外的**同一自身查询中实际发生的物体位移**，
可为B生成器补充状态相关的功能信用，改善到达并维持可完成操作状态的能力。
竞争解释是原表示/地址仍不足、普通动作学习已经覆盖了这项有效信息，或位移监督只形成旁路预测/想象而损害动作。
不把标签增加、梯度接通或预测变准本身当作前一种解释成立。

历史边界：G2/G3已有关系标签到LoRA/FM；VisibleObject已有实际读取监督及后期绝对收益；
LocalAction和Video Functional已有同表示/自身hidden辅助信用及负结果。旧物理J加权提案未过预登记近似前提。
本项不恢复这些配方，也不声称首次监督物理后果。新增的实际量是query自身对象的未来位移，
消费者是唯一LoRA的原生action-out额外坐标及随后action-in；不用时间平均Program、独立语义头或物理J。

## 2. 父点、参数和固定数据

- 父点为`conditional_read_write_continuation900_20261002/conditional_read_write/train/attempts/continuation/checkpoints/macro_00000900`，
  根在`/data1/user/ymdai/ember_runs/`；900实际续训身份85919994，原450父训练a0e0248d，Source1000原训练b8ea00e9。
  原完整bank、normalization、source、teacher与scene身份从`conditional_A_reexpression_diagnostic_20261002/Original/bank/panel_teacher{0,1}.json`读取。
  选择它是为了直接检验上述失败实例；不把其140/400当作超过成熟T161或强MT153，也不据此选择后继完整训练底座。
- 冻结source、公共A0/B0、native、四层解释器和全部A生成头。仅更新38处B生成器的
  `b_key/b_delta/b_context/b_dynamic/b_out`，各条件共享同一套参数；不引入task/teacher私有参数。
  完整输出始终是原38对A/B，执行hidden会随生成B改变；冻结A不是对未来方法自由度的永久限制。
- 四个train task及teacher固定为0:(40,11)、12:(25,14)、20:(38,42)、32:(17,43)。
  精确复用`operator_learning_limit_diagnosis_20260930/query_manifest.json`的64步、每task每步28个query、frame及flow seed。
  两teacher共享其task当步查询，均排除这两条teacher episode；8条件各1/8，每臂14,336次condition-query使用。
  数据量没有扩大；B20是原2/28/20划分中的固定独立查询面板，不是官方held。
- 两臂从同一父权重开始，fresh AdamW：lr1e-4、betas(.9,.95)、eps1e-8、weight_decay1e-4、整体clip1，无scheduler。
  64个有效更新，seed7，完全相同的逻辑样本和32维noise/time。保存起点引用及64完整恢复状态；不按中间分数选点。

固定公共native/解释器/A允许复用同版本H/c/d、最终A、K及delta-z；随B头更新的Value/M不得缓存跨更新。
优先复用已有合法原件，缺少的8条教学native各读取一次；不复制大模型、不载入teacher state/action。
保持原完整B递推及有效梯度。可缓存不依赖B头的量以提高吞吐，不用参数拟合或历史bank残差查表替代编译。

## 3. 位移标签只来自已有query episode

标签为明确实体在world坐标的实际位移，不称抓稳、接触、意图、成功或通用状态。
固定实体由原任务语义与官方XML/BDDL确定，不按结果挑选：

| task | 位移点 | 类型 |
|---|---|---|
| 0 | `akita_black_bowl_1_main` | 正确黑碗body原点 |
| 12 | `salad_dressing_1_main` | salad dressing body原点 |
| 20 | `wooden_cabinet_1_middle_region` | 随中抽屉运动的site；不是柜体平均位置 |
| 32 | `moka_pot_1_main` | 摩卡壶body原点 |

task0/12/20的映射沿已审计operation_semantics schema；task32按当前XML解析实名body。
若当前资产无法支持上述精确物理点或观测时间对应，报告具体语义边界，不自行改实体/平均对象。
task32此标签不包含开炉信息；主动作FM继续覆盖完整任务，失去开炉能力必须完整报告。

令当前query为post-action `obs[f]`，第h个未来动作是`actions[f+h]`，h=1..50。
令p(i)为与`obs[i]`对齐的该物理点位置，固定

`r[f,h] = (p(f+h) - p(f)) / 0.1m`。

0.1m只是预先固定的物理单位，不从本批分数/held估计尺度，不改source动作或state normalization。
从既有HDF5存储state/XML做CPU运动学恢复，沿7bd8a6f5的`audit_operation_semantics.py`已核定的
`states[i+1]`及一个2ms积分子步缓存对应；无环境step、扰动、渲染或新示范。
标签恢复仅限原A28训练池及固定B20 query所需的帧/未来窗口，不读两条teacher的特权state作为条件或监督。
保留明确frame对应、点名和有效mask。缺少真实未来观测的位置不计位移loss，不重复末帧冒充真实效果。
主动作50×7及原末动作补齐规则保持；位移无效位置的clean target填0、loss mask为0，并记录其数量。

标签、实体表、state、future frame和有效mask均不能进入Writer/native/Compiler条件。
标签只在完整LoRA生成之后构造训练query的带噪输出与loss；部署没有这些额外信息。
不得读取官方Val/Test动作、state或reward；没有新数据来源或在线状态收集用于学习。

## 4. 唯一学习对照与真实消费者

仍使用原32维action suffix、50个horizon、官方十步Euler与同一source算子。以0-based维度记：

| 臂 | clean flow target | loss |
|---|---|---|
| A（action） | 原`[a7, 0_25]` | 原完整50×7动作FM |
| J（joint） | `[a7, r3, 0_22]` | 原动作FM + 有效位置的3维位移FM |

两臂对0:7维使用相同真实a、noise、time及权重；J将7:10维赋予上述位移含义，10:32维仍沿原未监督padding。
policy的环境action定义/processor仍为7维；只修改内部已pad的训练target，不把位移注册成新增环境动作。
不额外增加zero-padding loss、动作权重、端点/前5 loss、公共FM或蒸馏；新位移项权重固定1。
若某query没有有效位移位置，其位移项为0；否则先按该query有效h×3求均值，再按query/condition原权重平均。

对J令`y=[a,r,0]`、`z_tau=tau*epsilon+(1-tau)*y`、`u=epsilon-y`，实际

`L_J = mean_50x7((v_a-u_a)^2) + mean_query mean_valid_hx3((v_r-u_r)^2)`。

完整源计算均可微于生成LoRA；两个余切共同回传同一38处B及其五组生成参数。
source/A0/B0/解释器/A头不更新，也不误把对label或查询hidden的反传当作Writer条件输入。
label进入带噪query输出是普通FM训练接口，不能把tau<1的hidden回灌教学编译器。

部署时A/J均从同一32维Gaussian噪声开始，用原十步更新全部32维；只把前7维去归一化并执行前5。
J的7:10维预测会被下一步原action-in读取，输出仍由唯一LoRA形成，无额外预测器、reward或在线优化。
该读取关系可能被学成弱作用或无用作用；本项不预设未来坐标对动作有必要贡献。
其余22维没有完整联合密度监督，因此不能把理想低维joint-FM一致性定理直接当成此32维采样器的保证。

## 5. 固定终点的行为和功能读回

每臂64后用原四task×两teacher×init0..3，共32行；合计64新增环境行、16个不同物理初态。
原当前900的`Original`32行直接复用为父参照；旧T的parent/S/P/D仅作另一个父模型的历史背景，不混为匹配臂。
无official400、相邻选点、Test、wrong/shuffle/reverse、扩teacher或结果驱动补例。

每臂六full：task0/12/20各第一teacher init0；task32 teacher17 init2、teacher43 init2/3。
合计12 full、52 compact；全体保存goal、continuous、真实执行actions及完整50×7 action chunks。
双256→224、双相机旋转、state8、source normalization、10 flow、前5、原scene/settling10/root7和绝对policy噪声时钟均保持。
按canonical dynamic long-first queue与persistent worker运行，成功即停。

J另被动保存每个replan的完整50×3位移预测，以及该点的每个实际执行步位置；不增加forward或决策分支。
只把已执行的1..n（n≤5，可能因成功提前停止）预测与实际位移比较，不把未执行50步的未来当作已验证物理后果。
按episode/teacher/task等权报告米制RMS、相对零位移预测风险及真实运动能量；静止、失误与成功case全部保留。
它仍是当前行为分布上的有限预测证据，不是任意动作的物理模拟器或抓持标签。

同批父/A/J在原固定A28/B20 query上读取动作FM及B20官方十步动作，保存全50后分别报告首5、全50、有效未来和motion/gripper。
J另报告位移FM和十步输出相对该query原示范位移的误差；生成动作未必等于示范动作，所以这项离线误差不是其真实动作效果误差。
父原A28结果可复用；缺少的固定面板读一次，不新拟合探针。保留全部逐query值和不利任务，不以离线分数选点。

## 6. 裁决范围与停止线

主要比较J对A的完整成功、每task/teacher、R/G/L/churn及父成功保持。
具体看task32是否增加实际搬壶和完整放置，并保留开炉；不是仅更早闭合、过3cm或改变轨迹。
还要看其它任务及两teacher是否出现实际获取，避免只凭一例或A退化形成的差额判断。

- J取得跨condition的有意义控制收益、能力交换可接受，且物体预测在实际执行上有相应依据：提高此学习联系的支持度。
  仍不证明增益来自flow内反馈而非共有表示学习，也不证明未见任务/视频迁移；不自动启动完整fresh。
- 预测更准但控制无益/更差：所测未来效果信用未解决完整调用，不追加λ/坐标/层位或训练时长扫描。
- A已改善而J没有额外收益：降低“这类显式效果信用是当前主要缺口”的优先级，保留普通FM可学正例。
- 两者均无有效改善或位移未学好：只说明固定表示/A、当前B函数类与64步条件下没有取得依据，
  不宣布所有效果监督不可能；也不以这项范围限制自动续训或扩大函数类。

所有分支均在固定64及读回后停止新增计算，完整报告实际学习程度、行为正反和解释边界。
没有新canonical架构、RL、dataset扩展、追加seed、层/权重/scale搜索或自动后继。

## 7. 资源、实施和交付

唯一root为`/data1/user/ymdai/ember_runs/joint_action_effect_credit_20261003/`。
预计含工程、标签、训练、编译/读回**2–4小时**，硬限**4完整GPUh、12GiB新增峰值**，预计峰8–10GiB。
依据旧S64约1537秒/.427GPUh；两臂主FM加同forward三坐标信用约1–2GPUh，native/编译/全部读回约.3–1GPUh。
CPU标签约15–35分钟、工程约60–100分钟、并行学习/读回约30–65分钟；以实际profile修订预计，不能静默超预算。
达到3.5GPUh而剩余无法在4内完成时报告具体缺口；不靠删case或未记加载/失败成本维持预算。

执行者按现有约定独占工程与canonical tracked/Git窗口，负责实现、消费者检查、集成/push和clean detached冻结运行。
复用现有conditional/native/FM/evaluator owner，不恢复旧独立trainer或建立第二长期Writer运行面。
必要检查只覆盖实体/时间/信息墙、标签mask、两臂共同样本与主动作信用、完整38-LoRA/梯度、原生32维部署及成对场景。
不做逐bit、逐tensor或新hash审计。普通BF16/TF32差异按仓库合同处理，不为低位差异改科学计算。
保存实际标签来源/实现、参数和loss合同、起点及64完整恢复、bank、raw rows/预测、汇总、失败和完整费用；不复制source或数据。

启动前live检查两节点GPU、data1独立quota/个人实占/共享容量；两臂及读取按真实吞吐并行，遵守仓库总物理卡数规则。
有显存余量须验证更大microbatch/frame chunk或合适并行，保持有效batch、condition权重及64步语义；不为填显存添样本。
profile只使用登记输入，最多两臂各两次可丢弃更新后恢复同一初值/RNG，不新增环境smoke矩阵。
正常长任务一次等待退出，不轮询进度；整批一次完成或实质边界回报。
结束后退役专用入口/hooks，保留Git/frozen/原件，推送并交回canonical窗口；不追加agent或自动自通知链。

### 完成后的来源标注纠正

main在§118登记前直接核对900的`launch/code_identity.json`和实际`run_contract.json`：
900续训/原correct400读取为85919994，a0e0248d是450父训练；原32行诊断bank的`reading_git`为923ff89b、`training_git`亦为85919994。
本合同及§117早先把900训练/物化都缩写成a0e0248d不准确，现作上述纠正。
本批b0df34ee学习实际读取的900权重路径一直正确，不改变任何模型、样本、原分数或成本。
已封存的运行合同/identities原件不覆盖；后续来源须明确列出这条纠正及原始完整链。
