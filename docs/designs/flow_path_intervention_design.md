# 冻结flow指导位置干预：首步与实际中间latent

2026-09-26。是否active只看progress；机器合同`configs/flow_path_intervention_v1/experiment_spec.json`。
主讨论01a0cd94-65da-7b22-8ca9-7ba35f454632；执行Sol 01a0cd90-ebb7-77a1-a20b-a858825d2f66。
本批32个保存query的冻结计算＋64条诊断闭环，无训练、梯度、新Writer前向或新bank。

## 1. 问题、历史约束与信息价值

§150没有兑现探索目标/条件迁移解释，§147–150有限回报更新分支关闭；不从这项阴性推出FM错误。
新问题来自更长历史：普通FM、真实十步endpoint蒸馏、learner环境状态的expert velocity监督各有有限正例和保持失败。
[完整复核](../analyses/flow_supervision_history_20260926.md)区分了SEOD/GOMQ的端点反传与phase aggregation的插值latent；
不能把本批叫“第一次使用专家/真实动作/学生状态”。旧时间轴检查也未发现简单的端点单独改善或A/B相消。

待测机制是：同一环境观测下，学生实际十步去噪访问的latent可能需要不同于直线插值上表现的纠正；
若该纠正对闭环没有用，即使场差很大，也没有理由开始按该分布蒸馏。
先直接换一次真实velocity调用，区分“初始条件场就需要改”与“中途状态上的纠正有额外价值”，避免先花费长训练。
同一完整冻结expert作为供体；首步/中步各只替换一次调用，完整expert检验供体本身是否在本面板有用。
不是把第二expert引入部署，不把小诊断当EMBER能力成绩，也不声称这批证明视频的必要性。

选择global2（bowl table-center→plate）与global12（salad dressing→basket），均为当前合法fit任务。
前者当前P原条件4/4、旧expert41/50，作已有能力参照；后者P1/4、旧expert46/50，作获取空间参照。
选择利用了历史成绩，是定向诊断而非盲测。旧成绩不代替新对照；两个任务不外推至held或全部40任务。

## 2. 数学对象与竞争解释

固定观测o、精确语言、完整LoRA及初始Gaussian epsilon。十步为t_k=1−.1k、h=.1：
`x_(k+1)=x_k−h v_S(x_k,t_k,o)`。S为当前student，E为合法task expert。
`F^S_k(x)=x−h v_S(x,t_k,o)`，从第j步替换一次的有限端点作用为

`Delta a_j = P5 [F^S_9 ... F^S_(j+1)(x_(j+1)^S−h delta_j) − x_10^S]`，
`delta_j = v_E(x_j^S,t_j,o)−v_S(x_j^S,t_j,o)`。

P5取真实前5×7。小扰动一阶项为`−h P5 product_(k=9..j+1)(I−h D_x v_S,k) delta_j`；
本批直接计算有限作用，不把线性近似或场MSE当闭环成功保证。
两条完整采样路径还满足精确分解

`x_10^S−x_10^E = −h sum_k [v_S(x_k^S)−v_E(x_k^S)]
                         −h sum_k [v_E(x_k^S)−v_E(x_k^E)]`。

前项为同一student latent上的场差，后项为expert对路径偏移的响应；两项可相消，不能只按能量分配失败比例。
同时比较直线`z_k^A=t_k epsilon+(1−t_k)a_A`，A取S或E的本query完整50×32端点。
测`D_path=mean||v_S(x_k^S)−v_E(x_k^S)||²`及`D_line_S/D_line_E`，k=1..9、全50×7；k0单列。
两个插值锚避免把phase历史的student-action插值与expert-action插值混称相同分布。
这些是**相对某个冻结expert的场差**，不是普通FM风险、真实动作误差、概率密度比或正确语义量。
没有密度比/光滑性界时，小插值风险不保证实际路径风险；但两者不同本身也不证明训练根因。

- **H_midpath_useful**：E在获取任务确有优势，H5比S与H0得到更多成功；路径场差高于两种直线且不只由一个query/夹爪维度主导，
  才支持“中间latent的指导值得后续学习分布对照”。当前任务较少，只能是局部依据，不是统一根因。
- **H_early_condition**：H0有益而H5没有额外益处，或场差从t1起就相近；不支持特指中间路径覆盖不足。
- **H_broad_or_coadapted**：完整E有益、两种单步都无益/有害，说明单位置替换不足或两场不协调；
  不能据此否定所有expert学习，也不能直接投入路径蒸馏。
- **H_reference_unhelpful**：E在新面板没有优势，历史expert不能校准本解释；不换task/供体/step追逐正例。
- 仅D_path较大、端点接近E或几何更近而闭环无收益，均不支持修复。H5与H0还改变了纠正被后续积分传播的长度，
  因而有益时间位置不等于已经唯一识别训练覆盖原因；只有后续同标签/同预算的学习分布对照才能进一步裁决。

## 3. 固定模型、来源和信息墙

S=C_S00@1155，父训练7dc95edb；使用dd2e00bc实际采集bank，不重新编译：
`/data0/user/ymdai/ember_runs/return_credit_direction_causality_20260925/banks/collection/P/`
下`task_002_teacher_34/bank_record.json`与`task_012_teacher_38/bank_record.json`所指完整rank16载荷。
S基础模型：仓库`runs/outputs/pi05_source_aligned_seed7_1k_20260915/checkpoints/step_00001000`。

E必须独立使用旧基础模型`runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000`，
加旧bank根`runs/outputs/pi05_task_expert_bank_formal_step1000_r6_81101fe_20260807`下：

- task2：`worker_1/task_01_global_02/checkpoints/step_00002000/adapter.safetensors`；
- task12：`worker_0/task_06_global_12/checkpoints/step_00002000/adapter.safetensors`。

旧source/expert训练offset0，当前source/Writer训练offset1；不把旧adapter移植到aligned base，不按当前任务重训expert。
原normalization与当前内容相同、tokenizer与执行预处理相同，动作坐标可比较；base权重不同，必须保留各自prefix KV。
历史expert不是完美oracle，它在当前student状态和latent上可能错误，故需要完整E新闭环参照。
每个worker可在同卡持有两份冻结模型，显存按实测准入；不复制基础资产，不要求两张卡服务同一episode。

只读两个task的旧权重、保存query和当前bank；无新teacher action/state/reward/terminal标签读取，
无held expert、official Validation/Test或参数更新。保存query含执行者自己的观测/state，不是部署Writer的teacher输入。
E明确是privileged诊断供体；完整E不声称使用了视频。S/H0/H5记录student视频34/38，E记录自己的expert身份。

## 4. 32个冻结query的计算矩阵

原dd2 study的`collection/groups/task_{002,012}_state_{00..03}/replica_0_decisions.pt`，
每文件全部四个reservoir decisions，共32；已CPU核对Q/M分别为task2的20/4、20/4、19/4、21/4及task12四个56/4。
不按原success/credit或新差值选query；沿原replan index保留身份、raw_input、processed、flow_noise、mu_old_full。
用原输入/原noise，S读取实际bank；E从相同raw observation按共同processor独立建立旧base prefix。
不重放环境、不增加执行观测或噪声seed，不给E使用S的适配prefix缓存。

每query保存四条完整50×32路径S/E/H0/H5的x[0..10]、v[0..9]；
另存E@S路径、S@E路径，以及S/E在line_S和line_E上的全部velocity。
每query最多100次denoise调用、32×100=3200，等价计算不重复；共享t1可以复用但要记录实际数。
保存规范化与环境坐标full50×7端点、实际前5与夹爪sign；无需重复缓存RGB或视频特征。
精确分解、逐time/维度差、两种line差在CPU计算；按task→初态等权→四query等权，不把长episode的Q当权重。
只报告该固定replica0的保存状态，不称总体occupancy无偏估计。

第一个task2/state0/最小replan query为pilot并计入32。
最多额外40个denoise调用只用于该query上S/E相对canonical endpoint各一次、H0/H5供体=S的no-op各一次；
没有额外环境smoke、query或重采样。最大GPU计算为3240个离线denoise，闭环自己的正常调用另计。
native比较前5归一化RMS≤.01、finite/shape正确、无Source梯度、旧bank身份明确；
若实现和精度正常但干预信号与复算差同量级，报告未识别，不追逐bitwise或改dtype。

## 5. 四格64条闭环

两task×states0..7×四格，每格16、总64；S/H0/H5在同task复用同一视频条件，有限训练诊断不声称50视频无放回。

| 格 | 每个replan的十次velocity来源 |
| --- | --- |
| S | 全部当前student |
| E | 全部旧base＋对应expert |
| H0 | 仅k0/t1使用E，其余九次S |
| H5 | 仅k5/t.5使用E，其余九次S |

混合格在**本格当前latent**上查询E。替换完整50×32 velocity后继续原Euler，绝不拿E自己路径的latent/动作替换，
不只换前5、不截断horizon、不插值参数或混LoRA；E/学生各使用同一个真实当前观测及自身base的prefix。
每replan只有十次velocity调用，无额外expert动作采样。首步与中步均用一次供体，未调混合强度或time。
H0/H5额外旧base prefix不消耗policy RNG；四格初始noise的完整seed schedule相同。

除该显式诊断干预外，canonical render256/model224、双相机rotate180、state8/action7、flow10、前5后replan、
dummy10、horizon Spatial220/Object280、成功即停、env/policy seed7不变，无探索噪声。
继续用同一rollout engine、cost-balanced queue与persistent workers；不建立并行模拟器或重写策略评测。
复用现有底层denoise/被动capture owner，新增scope只允许本研究两task/两bank/四格，不放宽旧研究入口。

task2/state0四格为pilot，计入64，只按接口/初态/种子/动作/trace/退出验收，不按成绩放行其余60。
所有64条保存compact、实际动作、T+1连续对象/fixture/EEF/gripper/全部stage predicates；
full固定两task×states0/4×四格=16条双相机。
每条闭环只旁路保存首replan的完整x/v及供体调用身份，无新增函数预测；其它replan保留调用计数、time及来源。
同种子列表只比较非空共同前缀，同时逐条验收各自全部公式；成功即停导致长度不同是正常行为。

## 6. 读出、判断与停止

预定简单差E−S、H0−S、H5−S、H5−H0、H0−E、H5−E；每task及固定两task合计均报告成功数、R/G/L、
churn/Jaccard、breadth和完整成功集合。20000次、seed20260926；每task独立重采八state，四格始终配对。
两个task固定，不把二任务bootstrap伪装为40任务泛化；各差复用同一采样，报告95%描述性区间。
离线读出按四state配对bootstrap，保留逐query/time/分量，不能用3200次forward冒充3200独立样本。
所有环境几何只作事实，不自动解释为语义理解或抓取成功。

H_midpath的一个小样本正点值不是批准长训练的充分依据；需同时看完整E的可用性、H5相对首步对照、
原student成功保持和场差是否与叙述一致。任何结果都由主讨论回到原件裁决，Sol不按排名选择下一阶段。
64/32完成即主动Queue主讨论并停止新增实验；不追加位置扫描、步数/强度/seed、更多task、专家训练、Writer学习或held评测。
若E无优势或两种单步均无收益，保留阴性，不重新挑任务/模型来完成期待。

## 7. 实现、预算与原件

Sol从最新main隔离实现，验证后集成push，所有新GPU计算统一clean pushed detached commit F。
旧dd2 bank/query、7dc训练、旧expert/source均只读，来源跨历史本来不同，不要求重生成来凑同commit。
新F只含冻结诊断和必要scope/采集/分析；遵守结构gate，不恢复历史trainer或通用新框架。
如仅CPU验收/调度/分析有错误，可在独立clean pushed修复提交完成其CPU职责，明确绑定实际GPU F原件；
不得因此重跑有效闭环、热改F或把修复提交写成旧GPU来源。若必须改变GPU计算行为，停止受影响执行并报告主讨论。

study `/data0/user/ymdai/ember_runs/flow_path_intervention_causality_20260926`；
预计.7–1.3 GPU-hours，**硬限2 GPU-hours**覆盖模型加载/初始化/工程/失败/离线/闭环；最多同节点两卡，项目≤6物理卡。
新data0≤3GiB，开发＋formal代码≤768MiB。Sol开root前查strg01独立quota/共享容量和实际峰值，
每GPU launch同时live检查gpu01/gpu02身份/所有权/余量。正常长任务一次持续等退出，不轮询cache/log。

保留精确命令/env/设备/完整进程计时、两base和四adapter实际引用、32query索引与预测、64原始rows/trace、16cases、
工程pilot核验、每worker退出及completion；所有要求产物/缺项如实列出。
预计或实测超限、OOM、源身份错误、干预调用数/输入不符、非有限或输出缺项时停止受影响阶段并主动报告。
只在实际未解决工程问题上修复，科学不通过不触发重跑。
