# 自身功能修订 Compiler：完整方法与首批实施合同

日期：2026-10-10。科学决策由 main 负责；这是 Owner 认可完整推导后继续推进的候选，尚无新方法性能证据。
main已接回c60882e1、消费[完整诊断](../analyses/parameter_edit_credit_diagnostic_20261009.md)及实际配对/行为原件。
本合同登记完整方法，并授权§5的实施与有界实测；不是正式长训/held评测启动记录。旧诊断的结果、成本和封口保持。

## 1. 实际问题、解释及这次取舍

当前参数 Compiler 从 MT153 出发，完整 χ180/360 为127/130；train48的MT/end/null仍为27/25/25。
旧keep只把同一个专家FM方向乘1或1.2，没有独立的自身功能保持方向。
新的固定incoming定向面板中，I/E+/E0均13/24；I→E+ R11/G2/L2，E0→E+ R8/G5/L5。
因此不能把经验简单删除，也不能由一次自身成功推定最后一次编辑或新初态一定有益。
在这8个condition的固定自身观测上，E+−E0的前5动作均方变化远小于I→E+；少量差异仍伴随成功集合交换。
这削弱“当前已学到稳定有益的经验定向修订”，不证明经验没有作用，也不将小动作差异等同于微小闭环效应。
新旧执行的正常数值差异与有限定向面板边界保留；不据此开逐bit复现或再跑翻转样本。

本次撤换raw A/B编码、rank-token编辑及64标量chunk decoder。
共享网络负责从视频和真实经历判断当前控制响应应受到什么修订压力；当前policy的真实导数负责把压力写回完整A/B。
训练必须同时具有密集功能信用和实际修订效果信用，不能只改变参数化后仍宣称解决了经验学习。
这是完整方法的组合干预，不把结果归因于其中某一独立模块。

历史约束：PPW已经做过预测功能信号、真实导数回写和外层FM，合法闭环弱；Local Correction Field已有局部真实余切监督，也未学成。
最近似完整原理见[PPW](process_pullback_writer_design.md)、[Local Correction Field](local_correction_field_design.md)。
T的161/150/public103保留其有限视频正证据；ADSP、SKNC等保持约束并未阻止其它新初态丢失。
本次实质区别限于当前实际LoRA、自身真实状态的十步响应、动作及实际后果、修订后新初态回报这条完整联系。
梯度形式本身不构成新成功证据。若只改善内部误差或已访问状态，必须降低相关完整假说的支持，不能换名字续投。

## 2. 继承的数据、来源与部署边界

沿用coverage_v1显式24/8/8，以及已审计36个等权meta tasks：
`0,1,2,4,5,7,12,13,14,15,17,19,20,21,22,25,28,29,32,34,35,36,37,38,42,43,51,55,56,62,64,73,95,96,97,101`。
canonical source step1000、coverage MT step300、tokenizer、冻结normalization、38 targets和实际rank128映射均复用原资产。
source及原生图文prefix冻结；最终始终只有一套完整rank128 A/B，执行`W_src+BA`，不叠加第二adapter。
K=1，指定同task action-hidden正确RGB视频、exact language、stride5及真实末帧。
教师action/state/proprio/reward/terminal/task ID/文件名/物体位姿不进入Compiler；不调held离线标签。
自身真实RGB/proprio/动作/后果和success等反馈按Owner最新许可使用；模型推断和真实观察明确区分。
最终参数固定后，Writer/导数修订/经验读取全部退出；condition内实践和最终新初态评测分开。
部署共享参数φ冻结，不生成正确动作轨迹，不拟合收集的动作，不建立每condition policy optimizer。
这里的梯度是共享学习过的编译算子；不是以构造轨迹为标签的task-local策略微调。

## 3. 完整编译计算

### 3.1 实际输入和保留的读取器

教学继续采用冻结原生双相机Φ（512×2048/时点）及合法固定MT probe H（50×1024、time1、seed1729、无teacher state）；
保留8空间slots/帧、width256、4层有序时序编码、语言/帧/camera位置和首末标记。
经验继续使用实际decision的pre/post Φ、当时真实H、proprio、实际动作及mask、反馈和episode/时间身份；
保留逐decision事实编码、2层有序编码和16个全局slots。历史H只作已发生的证据，不能代替当前policy的计算图。
教学空间query由language和经验改变，删除raw参数token的读取。以上共享模块全部fresh，不继承χ360权重。

从累计真实decision按时间均匀取M=min(16,N)个不同观测（floor-linspace，去重）；保留原policy-noise seed。
这16是首批功能支撑样本预算，不是方法永久上限；完整E仍参与读取。不按成功、误差或人工阶段挑支撑点。
每个支撑点的事实表示读有序教学memory及全部经验slots，形成c_i∈R^256；复用现有attention owner。
支撑观测必须经过真实Pi05LiberoProcessor，包括其自身8维proprio的冻结归一化/原生prompt，不能走无state teacher路径。

### 3.2 当前真实控制响应与学习的局部准则

以实际完整incoming Λ，在支撑观测和原noise上推进完整50×32 latent的10次原生flow；然后取前5×7的归一化动作a_i。
定义a_i=F_Λ(o_i,ξ_i)；不提前裁剪latent，不以time1 velocity、历史H或旧动作替代F。
c_i由停止梯度的原始经验/冻结特征产生，其计算不读取可微a_i。
能量头为`concat(c_i, Linear35→128(a_i)) → Linear384→256/SiLU → Linear256→128/SiLU → Linear128→1`。
末层无bias且权重零初始化，其余正常fresh；因此初始q_i=∂C_i/∂a_i为0，输出精确保留incoming。
零点不是永远断图：首步通过q_i对末层权重的混合导数学习，之后普通共享读取通路打开，须用真实消费者验证。
总准则C=(1/M)Σ C_i。C没有逐步正确动作标签；它由“这次修订产生什么策略”教会。
可微a只进入小MLP，使对φ/a的混合求导不要求在教学attention中构造二阶图。

### 3.3 完整A/B写入与真实外层信用

令每个实际MT因子的固定RMS为s（floor1e−6），沿用已存在的数值单位，S为对应逐元素对角缩放。
在无量纲因子坐标x=S^{-1}Λ做单位步修订，等价于

`Λ_out = Λ_in − P (1/M) Σ J_i^T q_i,  P=S²,  J_i=∂F_Λ(o_i,ξ_i)/∂Λ`。

q的幅度由共享能量头学习；不另开step-size/rank/seed扫描，不对因子更新做SVD/投影/融合。
P是明确的固定坐标尺度，不是真正natural gradient，也不宣称A/B gauge不变。
全部38个A/B共同输出；既不把MT冻结成另一个部署分支，也不单独限制B侧。
固定incoming、E、P与支撑点，不对历史环境、行为分布或此前编辑反传。
如果外层实际LoRA信用为v=∂L/∂Λ_out，则传给能量头的余切为

`qbar_i = −J_i(P^T v)/M;   ∇φ L = Σ (∂φ q_i)^T qbar_i`。

这在数学上只需当前policy一阶Jacobian，不需source Hessian；实现仍须验证十步VJP、JVP伴随、cache/hooks/checkpoint组合。
当前action_chunk带no_grad，不能直接假装可训练。NativeVelocity和batched LoRA底座可复用；旧8553b613的逐flow checkpoint、
85ecfd18的pullback伴随只提供实现原理，不恢复旧平行运行面。checkpoint重算必须重新绑定正确的完整因子。
不得以有限差分、单步velocity、detach内层梯度或假梯度静默替代上述合同。

局部推导`ΔF_new≈−J_new P J_E^T q/M`明确了限制：自身支撑状态提供的修订未必迁移到新初态。
这不是任意完整参数编辑器的等价表达，也没有把有限状态上的保持变成全局保证。

### 3.4 条件内循环

首批沿用MT初始化、1024真实环境steps（含settling）预算、episode结束/真实截断的编辑切点；这些都是当批比较安排。
累计有decision后，完整读取V/E并完成一次上述编辑；再根据刚才的自身success或预算耗尽决定冻结，否则新初态继续实践。
成功经验先被消费，模型可以学习零修订；不加硬编码的“成功后一律禁止改”。没有新decision的settling尾段不伪造编辑。
条件初态流、RNG、reset、tail和最终state隔离沿用canonical合同；没有固定J或次数菜单。

## 4. 共享训练的完整定义

### 4.1 真实事件、监督及保持

事件输入是明确的V/L、实际incoming及其累计E；旧事件可复用，但当前F/J必须重算。
第一段fresh共享监督使用同task、跨teacher episode的7×4=28个native FM query；保留offset1、50×32/前7维和现有padding语义。
优化`L_FM(Λ_out)+D_keep`；D_keep另从实际成功episode的真实decision均匀选至多16点，
比较incoming/outgoing在相同自身观测/noise的完整十步前5×7归一化动作MSE，与写入支撑重合时复用合法计算。
按这些保持点与35坐标平均；没有实际成功episode时为0。比较的是incoming函数，不把实际执行动作改作拟合标签。
系数1对应相同归一化动作的均方单位，是软目标而非不遗忘保证；旧ADSP/SKNC的负证据保留。
源参数/incoming不优化；所有共享V/E/能量模块通过真实Λ_out训练。没有原来的1.2 FM门槛项。

任务按36等权。正式学习批次须登记MT与实际非MT事件的独立采样层及覆盖，不能让endpoint数改变任务权重。
旧144个MT条件与72个refresh条件可作为合法bootstrap，不能把它们称为新模型自身经历。
继续训练必须包含新模型实际修订后的参数/经历刷新；不能再次用“各池一半”掩盖只有81/1440非MT曝光。
不伪造非MT incoming或给其配不属于它的历史；缺失覆盖如实报告，具体刷新规模随首批成本形成正式合同。

### 4.2 实际回报信用：原生十步均值的动作扰动

密集监督形成单一checkpoint后，接独立共享RL；fresh RL optimizer/scheduler，首个批次不混FM。
该段学习对象仍是共享φ，不是每condition LoRA optimizer；D_keep保留其软功能约束职责。
当前整条适应产生实际event/incoming/E；一次更新保持φ版本固定，信用只穿过当前编辑，明确为事件semi-gradient。
每事件使用2个与实践、固定train/final面板隔离的新初态，采集outgoing实际query结果。
query控制器的均值μ就是原生完整十步F；每次重规划在前5×7归一化动作加独立N(0,0.1²)，然后走官方反归一化/clip/gripper消费。
保存ξ、未变换的高斯动作y、实际执行mask和return。未执行后缀不进入score；没有把ξ的固定密度算成参数梯度。
该核比另换flow过程更直接对应正式十步均值，但训练有动作扰动、正式没有，仍有需实际裁决的差异，不能宣称完全相同。

logπ的因子余切由`(y−μ)/0.1²`通过完整十步μ传递。
每episode如需reservoir16个replans，按真实N/min(16,N)补偿；对2个query及逻辑事件平均，不能按GPU数、固定16或有效动作数改任务目标。
只用实际success回报；不使用隐藏的正确恢复动作、额外形状奖励或模型假回报。
incoming在同一新初态另跑独立policy/exploration RNG，作为停止梯度的return基线；不与outgoing共享采样噪声后把相关回报当合法baseline。
它降低状态难度方差，不凭空增加“哪里改对了”的标签。双方都无回报变化时，PG可能没有有用方向，必须如实报告。
RL目标为最大化新初态return减D_keep；没有value网络、PPO历史轨迹重放或环境可微假设。
外层PG得到v后，按§3.3回到完整能量头/读取器。最终评测的任何反馈都不进入这条梯度。

### 4.3 正式学习、读出及解释边界

本文件先冻结完整方法和下面的实施/实际profile范围；正式学习步数、真实刷新规模与检查点在获得真实VJP/JVP/PG成本后，由main另行登记。
这不是等待Owner批准，也不以机制smoke作为科学资格。实现者须实现完整训练消费者，不只交一个VJP演示。
正式节点必须采用single-checkpoint paired400、相同50-video调度、全部原始行；学习到有信息量节点后及时做完整读出。
先判断训练任务的净修订收益/损失，再判断held迁移及相对MT153/T161/旧135/当前130的完整得失；不能用train小面板选正式模型。
相邻节点/视频controls/科学停止在正式合同预注册；首批不读Test，不由shuffle/reverse选模型或改架构。
本方法仍可能只学公共修订、忽略视频、只改善已访问状态或在PG稀疏时学不动；结果须更新这些支持度，不能以公式自然为由保护。

## 5. 本次授权实施与有界实测

唯一工程owner继续是`src/ember/experience_compiler/`；替换当前model/decoder职责，真实flow/导数归execution owner，信用归credit/learning owner。
复用Runner、数据、缓存、scene恢复、persistent dynamic queue及LoRA执行；不另造simulator、第二Writer、兼容开关或全新训练栈。
χ360和旧诊断通过clean冻结commit/原件读取；消费后退役无复用必要的诊断入口和旧参数decoder，历史由Git及原件保留。
结构检查由实验session按code-architecture-gate自行完成；main不重复工程审查/测试。

实施先在独占codex分支完成，科学GPU实测来自clean pushed detached，保护当前所有原件。源/读出代码身份分别记录。
必要实测覆盖真实完整10-flow消费者、76因子VJP/JVP伴随、checkpoint重算绑定、零初始化后真实外层FM和PG到共享头/读取器、完整条件输出与资源退出。
验证应有小型实际方向导数/伴随关系的数值抽查，原因是这个新自动微分接口有具体正确性风险；不扩展为逐tensor/逐bit扫描。
工程消费者可以用针对性测试；不能仅用随机小线性网络或非零grad证明真实native链成立。

固定四个原有non-held条件：Spatial0/teacher29、Object3/teacher4、Goal0/teacher15、Long2/teacher44；
actual incoming分别为MT、MT、原incoming_001、原incoming_001，直接引用旧诊断run_contract及合法实际E，无新适应采集。
每condition的2个query初态取0–49中剔除其全部真实实践初态及原final32–34后的最小两个；运行前把实际ID写入新run contract。
query/baseline采用独立RNG域，输入支撑则复用各原decision的noise seed。
固定M16，比较物理support/native/query microbatch和checkpoint安排；逻辑算子、M、task权重、28条FM信用不变。
在同一完整事件流最多3个试运行optimizer updates（FM≤2、PG≤1，明确为可丢弃实现验证，不冒充正式学习）；不据其分数选择架构。
实际AdamW/LR3e−5/weight_decay0/clip1/constant保持现有量级；进入一次PG验证时fresh optimizer，不扫描学习参数。
最多16个新的query episodes（4条件×2初态×incoming/outgoing）；suite总horizon加settling上界5440steps。
PG消费者若需重复性能/导数检查，只能在那次optimizer step之前重放同一版本的score/输入；
不将重放充作第二轮on-policy学习，不重复采相同面板凑非零回报。
没有新validation/Test/视频因果矩阵/额外χ选点。0回报、无收益或不支持double-backward均原样记录，不扩大科学范围。

新ROOT：`/data1/user/ymdai/ember_runs/functional_revision_compiler_20261010`。
预计实现3–6小时，实际GPU检查/profile约30–90分钟；计算观察线2GPUh/3小时，新增峰40GiB，包含小有界frozen缓存、原始query、临时图和检查点。
已有无梯度72次十步约6.5–7秒、旧完整FM更新约10.4秒只能约束组件；新VJP/JVP和真实PG尚无实测，不能拿这些数字冒充新训练ETA。
分项记录视频/经验编码、F/VJP、JVP、外层FM/PG/keep、I/O、optimizer的实际wall/峰显存，并据端到端吞吐选物理batch。
显存明显有余量时继续有界放大，停止放大须有无收益/资源限制的实际证据；不是只交低显存数字。
launch前由实验session live检查双节点、6/8总卡和≤6单节点、共驻余量，查strg01独立data1 quota/个人用量/共享容量。
全部新增data1，旧资产只读复用；不复制大资产，不碰他人进程。第一批无需多节点拼训练。
超出上述范围或发现不得不改变计算/标签/更新语义时，给main具体成本与可执行取舍；正常接口修复在clean新冻结版本自主完成。

交付完整consumer证据、真实吞吐/内存、可执行正式训练成本分解、代码/资源/Git完成记录；一次可靠回报main后串行交回所有权。
main随后直接冻结正式学习与刷新规模并接续，不以Owner再次回复为条件，不把架构实现当作方法已经有效。
