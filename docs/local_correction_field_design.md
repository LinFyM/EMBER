# 局部条件纠正场：联合机制与有界算子核验

2026-09-14。整体goal及Owner自主授权持续；语义路径Writer与冻结回放均已关闭。
本文记录已经按§10关闭的联合研究假设。**§5的训练侧算子核验通过；§8–9的合法共同学习及八面板已完整完成，未获有益有序资格。**
当前状态由progress登记，历史正数和本文后续分支均不自行恢复运行。

## 1. 改变的是学习对象及其实际消费关系

冻结回放同时出现错物／实例、获取运输失效与组合目标失去，不能据此指定一个局部动作阶段为唯一根因。
旧Semantic Path的低阶有向统计加普通FM没有获得相邻过程优势；不保留该统计后加一个辅助头。
旧Native Correction则有实际纠正G的功能及有限闭环前提，但合法模型的多数参数误差仍在自身可表达空间内。
这些事实支持检验更直接的局部获取关系，不支持继续扩大A、调整体gain或恢复旧参数回归。

本候选学习：视频中某个具体状态及其已观察后续过程，要求冻结source在这个状态的原生计算中作什么纠正。
这些逐位置预测**本身组成最后的LoRA**，同时承担同位置真实纠正监督；不存在另一个只供loss使用的动作头。
这里的“纠正”是指定真实动作目标下的原生线性层输出cotangent，不是native Y、两次forecast差或阶段名称。
它只是候选的可执行知识表示，不声称包含视频的全部高层意义。

最近机制与差别必须完整保留：

| 近邻 | 已有消费与结果边界 | 本项具体改变 |
| --- | --- | --- |
| Local Action Grounded | 独立fresh局部动作头监督E，预测不进入LoRA；局部生成仍差于任务均值，闭环失败 | 原生纠正场既被监督，又直接形成参数，取消独立辅助动作出口 |
| Native Correction | 全局Q生成B，R由Q与裸X匹配；监督总矩阵BRX，合法模型近source | R在每个真实位置消费其有序视频上下文，监督同位置BR；参数由这个场与同位置X收缩 |
| EBSRI／PNBTT | 实际VJP作为共享生成器训练信用，前向值仍受native Y span限制 | 真cotangent作训练标签；合法前向只预测该场，不运行部署VJP，B不受Y限制 |
| 原生G诊断 | 真动作给出的G改善跨episode输出，闭环净+5.73pp；仍是privileged | 保留其条件×作用联系，先检验本项在局部场上限制rank的实际函数代价 |

`BRX`的代数形状不是新发现，Native Correction也可写成局部场的低秩形式。本项改变的是局部过程进入R的方式，
以及监督在收缩之前约束哪些位置产生哪些作用；不能把同一矩阵恒等式改名作为创新或成功证据。
原件见[可识别性§12](video_information_identifiability.md#12-固定source-prior与真实转移纠正近邻不同传递合同仍未成立)、
[Native Correction](native_correction_writer_design.md)、findings§87–94及[工作理论§9](temporal_control_compilation_theory.md#9-冻结回放后的解释边界2026-09-14)。

## 2. 完整前向机制

```text
exact language + 完整K1双相机RGB（stride5、真实末帧）
  → 原生图文语义和Action Expert完整50-horizon响应
  → 语言角色绑定的逐帧状态、完整视频双向且保序的上下文
  → 每个真实frame/horizon位置的条件过程表示e_i
  → 局部系数r_li和共同作用方向U_l，形成c_hat_li=U_l r_li
  → 与同位置裸source输入x_li配对收缩为唯一38-target LoRA
  → policy由自身观测诱导的x_l(o)触发相应纠正。
```

读取側VL／Action Meta与其它学习模块全部fresh共同学习，基础source始终冻结。
完整原生H在任务条件、跨帧learned read之前保留；动作horizon位置与视频时间分别编码。
语言角色先从真实图文tokens读取相关对象／状态，再结合全视频前后证据解释当前状态与后续操作。
参数作用的局部位置仍对齐当前frame和horizon，不能把全视频摘要广播后称作逐位置过程理解。

裸source X采用同组真实RGB、exact language、state-free prefix、public probe1729及t=1的固定只读读取。
它属于冻结执行坐标；观察Meta更新不会移动这个地址坐标。部署没有teacher actions、state、reward、terminal、task ID、
运行时loss／VJP、optimizer或任务更新。多阶段读取仍在rollout前同一次Writer调用内，执行时不再次访问视频。

设T为视频帧数，i=(t,h)，每层U_l为d_out×16，r_li为16维。定义

```text
B_l = U_l
A_l = (1/T) Σ_(t,h) r_lth x_lthᵀ
ΔW_l = B_l A_l = (1/T) Σ_(t,h) c_hat_lth x_lthᵀ
Δy_l(o) = (1/T) Σ_(t,h) c_hat_lth ⟨x_lth, x_l(o)⟩。
```

当前LoRA contract的alpha=rank，scale为1。该恒等式给出了明确消费关系：本机器人当前的原生激活与教学状态匹配，
决定哪些已编译纠正影响该层；下游冻结attention／非线性／10步flow把它转为动作。它不是教师时间播放。
U_l由视频条件产生，r_li同时消费同位置裸X与已解释的e_i；没有另一个自由A/B decoder绕过这一场。
不对生成A单独归一后丢弃范数，因为那会改变上述实际收缩。合法identity采用r出口零初始化、U非零，
第一步可学习r，之后U／读取器共同获得真实信用；不要求第一步每个模块均有非零梯度。

这个核由source既有状态表示定义，尚未证明其跨初态匹配总是正确。语义实例混淆、source表示缺失、
相似状态需要不同作用或多层相互影响，均可能使它失效；真实G正例只支持有限条件，不能消除这些风险。

## 3. 获取、迁移与保持怎样共同学习

只有跨episode主FM时，网络仍可能先学公共task修正。新增局部监督在授权训练视频自身RGB与动作之间建立对应：
对真实观察位置p，source的预测与其后已记录动作形成误差，标签为该误差在各原生线性层输出上的cotangent。
部署从RGB推断它；真实动作与导数只存在于训练专用目标构造。

在相同位置集合上，纠正场准确且按§2收缩，实际参数作用也随之准确：
`||G_hat-G||_F <= ||C_hat-C||_F ||X||_2 / T`。
这区别于一个A侧语义头在B=0时仍可完美拟合的情况。它不证明有限rank的场可拟合、不证明少数监督位置已覆盖全视频，
也不把该参数误差界当闭环性能界；§5先测这一rank选择的函数代价。

拟采用跨episode完整LoRA主FM，加同位置纠正场误差；不恢复总参数回归、空间边际KL、独立动作头或RL混合。
局部标签是冻结source下的稳定目标，直接约束实际参与LoRA的内容；与共同变化的功能教师相比，它减少一种目标漂移。
共享的位置读出复用原生状态／动作知识，学习的是状态与已观察过程到纠正的关系，不建立task到expert的字典。
这些是改善获取与迁移的理由，仍不能保证共享更新保持其它task。保持继续由相邻和跨视频真实行为验收，
不通过冻结旧错误输出、LoRA范数或逐bit一致性替代。

为使训练覆盖与完整前向一致，后续若激活学习，将在完整视频位置上均匀抽取少量局部标签，估计全位置场误差；
不能只监督四个固定位置便声称已约束整条视频。真实末段仅对已有未来动作评分，无未来标签的位置对应零纠正目标，
不伪造后续动作，不向模型传入label mask／terminal。最终采样、固定单位、loss系数、完整字段和约一小时节点，
须在实现profile及任何新学习分数之前另行冻结；当前不为全池构建大标签cache。

“source在教师轨迹上已正确、在新初始化仍失败”的反例继续成立。因此本项不把真实纠正定为视频价值的充分统计量，
也不把辅助场目标设为硬约束或唯一训练目标；跨episode主FM仍直接评价完整作用。如果这种软目标仍妨碍新状态能力，
应据行为关闭该组合，不能用场拟合好为其辩护。

匹配frame_set使用同一全部RGB、native响应、逐位置出口和独立fresh训练；没有视频时间、相邻关系或端点身份。
它依然有每帧内容及完整集合上下文，不被减帧、置零或只看单图。真实局部纠正标签在两臂相同；有序臂可用前后过程
减少对当前状态作用的歧义，这必须最终表现为有益闭环优势，不能由标签误差差距或位置编码证明。

## 4. 为什么先检验局部场rank，而不是直接再训Writer

旧G在总矩阵上作rank16投影；§2则要求整条局部纠正场C先有16个共同输出方向。两者可表达性不同。
即使G可用，也不保证`rank16(C)ᵀX`可用；普通的场MSE又按其自身输出能量选择方向，并未按X重新加权。
这项具体差别可用现有24task／四teacher和原动作query直接判断，成本小于新的联合训练。

本次不做RGB预测、参数学习、closed-loop、held或最终controls。它不是整套架构的先验成功证明；
通过后才值得投入合法获取，失败则停止这个局部rank16场及其监督／收缩组合，不改rank或加校准继续维护。

## 5. 当前激活的固定算子核验

- 只用train24、demo16–19，复用原G诊断每video四个固定p及同一真实动作15×7，共96条件。
- 同一裸source forward捕获全部38目标的X与输出y；基础权重保持冻结，同值临时叶仅用于训练侧求导。
  `L=mean_(4,15,7) (F0-(epsilon-a))²`，使用原已冻结记录的eta，定义`C=-4 eta ∂L/∂y`。
  C保留4×50×d_out，故`C_flatᵀ X_flat /4 = -eta ∂L/∂W`。
- 对各层C_flat作固定q=min(24,shape)、niter2、seed20260923的rank16 SVD。
  若`C16=P diag(s) Vᵀ`，取`B=V[:, :16]`、`A=(P[:, :16] diag(s[:16]))ᵀ X_flat /4`。
  eta不重估，不改幅度；输出固定38-target／76-factor、alpha=rank16的唯一LoRA。
- 与旧G相同的query42–45，每task16位置，每个teacher条件使用全部16位置；seed20260924+100*task+clip。
  query用自身真实双相机／state，分别计算t1和官方full10，之后才对真实15×7标签评分。共1,536组合、两读出。
- 复用原reference.npz和queries.json及完整source身份，核对模型、处理、位置／truth与RNG合同；不重算或重选source参照。
  原始C、LoRA、eta、位置、预测、原G差额、局部rank能量与所有条件均保存；eta=0或非有限结果不得被筛掉。
- 先对同task全部teacher和query等权汇总，再对24task等权。20,000次task-cluster配对bootstrap，seed20260914。
  **只有t1与full10的source−新算子改善95%CI下界均严格>0，且各至少两个suite净正，才通过本功能前提。**
  同时报新算子与旧G差额，不把“仍有正效应”写成已证明两算子等价或无损。
- 未通过则关闭本具体候选，不训练其RGB预测器，不扫描rank／SVD／eta／probe／字段loss，也不扩query或补closed-loop挽救。
  通过仅允许继续完成§2–3合法学习实现与预登记；不得将privileged输出计入整体goal。

一次真实smoke固定task0/demo16，仅核验捕获、收缩／梯度身份、finite、LoRA安装及两读出，独立目录保留。
正式96不重复整层dense梯度检查。单个一次性入口由`diagnose_local_correction_field.py`拥有，结束即退役，
复用现有source、数据、原子LoRA应用和查询合同，不增加第二训练器或评测器。

创建根前strg01现场：data0为93,869,632KiB、data1为1,067,519,024KiB，各soft quota1,073,741,824KiB；
共享空闲分别约1.6TiB／83TiB。现有data0研究根合计39,078,880KiB；本项data0新增峰值6GiB，
覆盖96份FP32 C约3.04GiB、96套LoRA约0.46GiB及原始预测／临时输出。data1源码／冻结worktree合计新增预算1GiB。
formal从clean pushed detached commit运行，launch前同时检查两GPU节点，并登记实际环境、设备、命令、日志与退出状态。

## 6. 后续行为裁决保持

若算子前提通过，合法Writer仍须独立fresh的ordered／frame_set完整比较；不得承接旧权重、选择有利teacher或混入专家adapter。
正式学习前根据真实profile登记两个相邻节点，每节点train96及validation400；实际曝光与所有采样字段严格匹配。
有序在两个节点均须相对frame_set及source取得可信有益增量，至少两个suite有序净正，且相邻正确行为、breadth及suite覆盖保持。
随后才补same-task-other、跨初始化及已登记强内容参照；single checkpoint选定冻结后才进行最终内容／顺序controls。
暂不额外强制145/400；Test与RL保持关闭。不能把算子前提、场误差下降或任一单峰当整体goal完成。

## 7. 固定算子完整结果与下一阶段

4f55968d clean pushed detached完成96条件／1,536独立query组合及两读出，三个worker与launch均exit0。
原source／eta／teacher truth／query位置、noise与原生38-target身份通过；全部预测finite，原始场及完整LoRA保留。
真实smoke的场收缩／weight-gradient相对差异.0006498，保持正常数值差异，没有为低位一致改变计算。

| 读出 | source MSE | 原G MSE | 局部场 MSE | source−局部场95%CI | 正task／suite |
| --- | ---: | ---: | ---: | --- | --- |
| t1 | .11977705 | .11351420 | .11351439 | [.00253443,.01103297] | 17/24；4/4 |
| full10 | .16493043 | .15574849 | .15574597 | [.00326525,.01693516] | 21/24；4/4 |

两项预登记功能前提均通过。新旧算子均差仅为约−1.84e−7／+2.52e−6，配对区间都包含零；
可保留两者实测作用接近的事实，不声称数学等价、完全无损、严格非劣或新算子优于原G。
该结果解除本具体局部rank16场收缩的函数可用性疑问，没有检验合法RGB获取、完整视频各位置、闭环或held迁移。

按正分支关闭一次性诊断并退役入口，原始运行代码由4f55968d及冻结树保留；不追加oracle闭环、query或参数扫描。
原件、逐task／suite／teacher、原G差额与机器裁决在
[READOUT](../runs/analysis/local_correction_field_20260914/READOUT.md)及同根`paired_readout.json`／`decision.json`。
整体goal未完成。下一阶段直接完成§2–3已选联合机制的合法前向、训练标签采样与同一场的学习接线，
冻结完整profile／学习合同后再启动；不重新把其它候选或独立动作头加入当前主线。

## 8. 合法共同学习实现合同（profile与学习前冻结）

本节将§2–3细化为唯一实现，不新增并行候选。保留原width256、8 heads、两组语言／时间blocks和两组rank4 Meta，
native50及全部38个rank16目标。原二阶路径统计与自由A/B出口退役。逐帧任务tokens先读真实图像，再以任务条件
读取完整T×50原生响应；两组语言RoPE／双向时间RoPE形成T×L上下文。语言位置保留原token位置；
有序时间位置为真实frame index／5，无序时间位置全零。每个当前frame/horizon的原生状态再读取该帧已经包含
全视频前后信息的语言角色，形成e[T,50,256]。不把同一全局摘要广播后冒充全部局部状态。

target／rank queries从完整角色上下文读取共同输出方向U；按原生输入／输出shape共享小MLP，输出非零初始化。
各目标的局部系数r读取同位置e、投影裸X及该目标的共同上下文，经共享MLP和零初始化rank出口产生。
裸X保持原尺度参与§2收缩，A不再行归一，也没有独立自由A。采用固定物理单位
`sigma=5.823084826577233e-6`，为§7全部96条件、38目标原始C逐坐标的全局RMS；
内部U为无量纲，实际`B=sigma U`、`c_hat=sigma U r`。该单一常数进入模型配置和checkpoint，
不读取task条件统计，不再根据学习或闭环分数校准。零r／非零U给出合法identity；第二步检查整条共同梯度。

teacher/action池及主FM沿用16–41，逐condition排除其teacher episode；每更新四suite各一task、各64个完整50-horizon
真实7维动作query、权重1/4。每个condition由独立局部RNG在其全部T个采样frames中均匀无放回抽`m=min(4,T)`位置，
seed由该condition已持久化query seed与固定20260914派生，不消耗主动作RNG，不依赖设备或worker次序。
字段监督位置只用于训练loss索引，不成为Writer特征。各位置p的真实标签为已有的`actions[p+1:p+16]`，
末段按真实剩余动作数计均方误差，无未来动作为零目标；不复制或生成虚构后续动作，不把mask或terminal送入Writer。

局部标签在线由同一裸source／双RGB／public probe1729／t1计算。令L_p为该位置已有未来动作的7维均方误差，
`C_lp=-eta_video dL_p/dy_lp`。eta只引用既有624条训练纠正构造的已冻结teacher侧记录，
不读取其query分数、不重新定幅，不构建新的全池原始C cache。源权重物理requires_grad始终False；
训练侧仅令公共noise成为求导起点，捕获原生线性输出cotangent，不部署VJP或优化。

`L_field = sum_l ||C_hat_l-C_l||_F^2 / (m*50*sum_l d_out_l*sigma^2)`；
共同目标固定`L=L_FM+.1 L_field`，为全部坐标统一的软约束，不逐video或逐层重加权。
字段loss直接读取构成同一A/B的U/r，且与主LoRA cotangent在同一参数版本重放，然后共同传回Action／VL Meta。
AdamW lr3e-5、betas(.9,.95)、eps1e-8、weight_decay1e-4、clip1、8步warmup保持，所有学习模块／优化器均fresh。
有序与无序使用同一字段标签及所有采样字段，独立fresh训练；不混RL、旧checkpoint、独立动作头或总参数回归。

profile仅验证最长完整视频的两次真实共同更新及一次完整推理：source冻结、完整帧／H／38目标、同一字段收缩、
第二次joint finite梯度、峰值与吞吐；profile权重丢弃，不由其loss选择上述科学参数。
据实际吞吐在学习前固定约一小时曝光的两个等间距节点，随后完全沿§6资格及后续分支。
现有video／factor拥有唯一Encoder和场收缩；裸读取／训练cotangent由一个小模块拥有，
supervised、sampler、runtime及materialization复用现有owner；不保留旧Semantic Path可执行分支。

## 9. 最长profile通过及固定50／100正式比较

c4c07bbb clean pushed detached在gpu01物理4完成task38/demo36的全部517原始帧／105个stride5采样帧，
两次真实共同更新及一次完整部署推理均exit0。Writer10,255,712参数，加两组Meta共11,804,000可训练参数；
source参数无requires_grad或残留梯度。主模型11项机制检查与相关运行／数据／评测／训练142项检查通过，
624个既有eta的task／episode与sealed身份均符合合同。

第二次最长更新31.346秒，显存allocated／reserved峰值34.285／35.459GiB；完整无标签推理8.299秒，
全部38个裸输入、76个finite factors、完整50-horizon及所有105帧保留。
局部系数、U、裸输入投影、图像读取、时间读取和两组Meta第二步均有有限梯度；
Action／VL Meta范数约4.71e-11／3.30e-9，信用较小，不能把图接通写成已获得有效知识。
profile不复用权重、不据两步loss修改字段单位、初始化、lambda、rank、LR、seed或架构。

依当前seed7真实sampler及原cost-balanced调度，三GPU的100次更新估计2,998.53秒，
另含初始化、三次固定train24无梯度动作诊断及checkpoint开销，按约一小时冻结**50／100**两节点。
两臂分别fresh100，各400条teacher conditions、25,600条主FM queries及每condition四个局部位置；
相同数据、field seed／ordinals／真实剩余动作／eta、主query及policy RNG逐行配对，只有有序时间RoPE不同。
诊断只在0／50／100，继续固定teacher46–49、actions42–45、每task128 queries；不用于选择模型。

每节点都完成train24×四init32–35的96行，以及validation8×50的400行correct面板，共八面板1,984行。
train每task只用46–49四个教学视频各一次，validation每task全部0–49各一次；跨checkpoint／两臂固定映射。
§6的有益有序、相对source、至少两正suite及相邻保持资格不变，完整面板后才裁决。
既有配对source train17/96及validation47/400仅在执行合同、normalizer、assets和逐episode RNG一致时引用，
不使用旧Writer成绩替代新参照。未获资格则关闭本组合，不续训／小扫或补未触发的other／最终controls。

当前代码结构净增长约406行，唯一新增197行correction owner负责裸读取及训练标签；
其余复用既有生命周期。shape／信息墙／完整目标验证中的分支保留在各自owner，未新增架构上限违规；
原路径统计与自由A/B出口及其专属测试已退役，历史代码和证据仍由Git与原运行根保留。
正式学习须从包含此profile／节点登记的clean pushed detached commit开始，并在launch记录实际两节点GPU与独立quota。
原profile、完整timings／gradient norms、退出和调度估计在
[profile](../runs/analysis/local_correction_field_writer_20260914/profile/results.json)。

完整视频生成的原生读取耗时占比较高；在正式训练结束后，以固定ordered100、task38/demo36做一次纯推理物理batch核验。
只比较完整视频frame_chunk=4／8／16／32的LoRA/s、峰值与finite／完整性，不读取动作、reward或环境，不比较行为分数，
也不要求逐元素一致。按最快且能放入当前设备的批量物化；OOM仅排除该物理batch，保留该次原始记录。
该批量通过既有materialization入口显式传入、保存到`materialization_execution`；Writer参数／训练合同和全部帧保持原定义。

## 10. 完整50／100结果与关闭裁决（2026-09-14）

50dafb05 clean pushed detached完成两臂独立fresh100，每臂400教学条件／25,600主queries。
四checkpoint完整；46aaf22d在完整物理batch核验后采用native frame chunk16，完成八bank与八闭环面板，共1,984条件／rows。
全部25曝光字段、12诊断字段及实际source、视频／初态／RNG、stride5末帧和38-target／76-factor身份审计通过。
24个最终评测worker及全部launcher均exit0，无失败重试；八面板累计评测墙钟17,987.69秒。

| 节点 | train96有序／无序 | validation400有序／无序 | validation有序−无序95%CI |
| --- | --- | --- | --- |
| 50 | 18 / 16 | 48 / 47 | [0,+.75]pp |
| 100 | 17 / 17 | 47 / 50 | [−1.75,0]pp |

同合同source为train17/96、validation47/400。两个有序相对source的CI为[0,+.75]／[−.75,+.75]pp。
有序相对无序净正suite分别1／0，且相邻48→47；§6三个资格均失败。
四组validation的breadth均3、Spatial均零；S/O/G/L依次为O50 0/5/42/1、F50 0/4/42/1、
O100 0/5/40/2、F100 0/7/40/3。训练breadth均7，没有形成新的广泛任务能力。
有序相邻保留41、新增6、丢失7，churn13、J=.75926；无序41/9/6、churn15、J=.73214。

固定动作FM0／50／100有序.153285339／.153149835／.152403202，无序.153285339／.153149911／.152400151。
末节点约0.58%改善，有序23/24、无序24/24task方向改善；这点有限学习没有变成有益条件控制。
当前最早明确失败的预测是合法共享获取，而非已经获得强能力后仅在held上遗忘。
§7的privileged函数前提继续成立，但它没有使当前RGB预测器、局部参数化与共同loss在登记预算内学出可用的纠正。

按§6／9关闭当前联合组合，不续至150/200或扫描rank、scale、seed、LR、loss来挽救本轮。
不把低churn、接近source的成绩或算子前提当成功，也不宣称已证明RGB信息缺失、某一个模块失效或普遍不可学习。
没有selected checkpoint，未启动same-task-other、强内容资格、最终wrong／no-video／shuffle／reverse、Test或RL。
原始记录不替换、不融合；全部冻结checkpoint和证据保留，整体goal仍未完成。

完整逐task／suite、相邻集合、来源和执行记录见
[READOUT](../runs/analysis/local_correction_field_writer_20260914/READOUT.md)，
机器事实与裁决为同根`paired_readout.json`／`bounded100_decision.json`。
后继投入先遵循[工作理论§11](temporal_control_compilation_theory.md#11-局部纠正场比较后的学习假设修正2026-09-14)，
本关闭记录不自行激活任何下一实验。
