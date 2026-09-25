# 冻结回报更新的目标对齐：原采集条件上的探索开关交叉

2026-09-25。是否active只看progress；机器合同`configs/return_objective_alignment_v1/experiment_spec.json`。
主讨论01a0cd94-65da-7b22-8ca9-7ba35f454632；执行Sol 01a0cd90-ebb7-77a1-a20b-a858825d2f66。
本批是固定两模型×探索开关的128条冻结诊断，无新训练、梯度或候选。

## 1. 已有结果和本次能改变的判断

§147的回报正/反方向在独立state/video上都没有超过父，不能据此声称当前图不能利用回报；
该批已事前保留H_exploration：梯度对应有探索的J_Sigma，评测却是J0，同时换了teacher与初态。
§148只去除执行等价类内的夹爪幅度信用，完整方向旋转33.1度，但两组初态一致性未改善。
§149直接执行同系数RAW/RB更新：P/RAW/RB为15/13/15；RB保留RAW全部13条、增加2条，
新增仅在teacher46，未恢复RAW丢掉的3条父成功。强“保持修复”预测未兑现，不继续长训或尺度扫描。

这些结果没有测过**同一个学习采样分布上的J_Sigma更新收益**。
继续补旧面板states34/35主要收窄独立条件J0的排名，不能消除上述目标混杂。
本批用原八个采集teacher与states0..3、一个全新探索子流，交叉父/已保存RB与噪声开关；
它能区分“此有限更新只在有探索执行中受益”与“回到所优化分布也没有可见收益”。
前者才给未来研究训练/部署目标对齐的具体依据，后者不支持继续这一回报更新分支。
只关掉噪声的主效应不是更新有效的证据，必须比较两个模型间的收益及其交互。

历史约束不变：早期source的J0/J_Sigma为19/22（120条）只证明探索可改变既有行为，不能替代当前模型更新的交叉；
旧RL profiles只有工程证据。§145单项监督、§146lookahead没有联合修复；更早私有Writer/自由A-B、
梯度隔离和缩头的反例阻止将本次阴性直接归罪容量、head或所有任务共享。
这条回报诊断不能解释监督视频学习的统一根因，也不替代正确视频对语言/static参照的必要增量。

## 2. 数学对象与竞争解释

令c=(task, teacher, initial state)取原32个条件，固定Source及native-flow seed schedule。
完整Writer产生theta=G_phi(language, video)，执行前缀均值为mu_phi(x,c)，最终环境动作为T(u)。
在每次replan，对归一化5×7动作加epsilon~N(0,Sigma)，
`Sigma=C_rho ⊗ diag(.05²,.05²,.05²,.05²,.05²,.05²,.10²)`，rho=.8，
时间相关只在本次五步前缀内部；不同replan使用独立子流。T包含既有反归一化与controller行为。

`J_Sigma(phi)=E_c E_epsilon R(rollout(T(mu_phi+epsilon)))`；
`J0(phi)=E_c R(rollout(T(mu_phi)))`。
旧score及其条件期望修正的理想对象是前者；不会因为使用相同成功标签就变成后者的梯度。
两者占据的状态分布也会随探索开关改变，不把此差异简化为某一动作的单次平滑。
若局部可微，J0沿理想探索梯度的一阶变化是`alpha <grad J0, grad J_Sigma>`，其符号不保证为正；
有限步闭环不依赖这个可微近似。令`B_Sigma(phi)=J_Sigma(phi)-J0(phi)`，则精确地有

`I = [J_Sigma(RB)-J_Sigma(P)]-[J0(RB)-J0(P)] = B_Sigma(RB)-B_Sigma(P)`。

本批估计Delta_Sigma、Delta0与I；不是估计真正梯度的余弦，也不检验所有步长。

- **H_objective**：Delta_Sigma为正、Delta0非正，且I为正；任务/成功集合支持该分化，才支持本有限更新存在探索目标收益而未迁移到J0。
  只有I>0、两项均退化，不算兑现。小样本区间若不能排除竞争解释，只称未识别。
- **H_condition_transfer**：同条件Delta0和Delta_Sigma均有正收益，而此前独立条件J0没有净收益，
  与teacher/state迁移问题相容。跨批历史变化和teacher/state仍混在一起，不据此唯一归因其中一项。
- **H_no_local_evidence**：两Delta没有可靠正收益，或只有个别任务交换；该方向即使回到原条件也没有继续训练的实证依据。
  剩余信用采样噪声、有限步非线性与Jacobian低位边界不能由本批互相分开；阴性不证明所有RL或架构不可行。
- **H_noise_main**：噪声对两模型作用相近，I接近零；这不支持“探索目标差”是本次更新表现的主要解释。
  宽区间不是等效性证据。

单一父、单次已采集信用、每条件一个新探索replica，只能提供有界机制证据。
新replica不参与旧梯度，但task/video/state为旧训练条件，不称独立任务泛化、无偏模型选择或部署分数。

## 3. 固定来源和四格

P为原C_S00@1155完整Writer权重，训练7dc95edb；RB直接只读复用34ea27bd的已保存候选weights，
不得重新更新、归一、插值或调整SGD系数。保留§147中actual-bank策略VJP/重建Writer Jacobian的低位差异边界。
Source1000、normalization、原C_S00配置、38-target rank16均不改；两个模型均冻结。

四格P_J0、P_JS、RB_J0、RB_JS。teacher逐task固定如下；这正是旧采集occurrence165的条件，不按成功挑选。

| global task | teacher demo | 原event |
| ---: | ---: | ---: |
| 2 | 34 | 4620 |
| 5 | 13 | 4622 |
| 12 | 38 | 4624 |
| 17 | 8 | 4626 |
| 22 | 5 | 4628 |
| 25 | 36 | 4629 |
| 34 | 2 | 4633 |
| 37 | 45 | 4635 |

每task固定states0/1/2/3，各格32条，共128；不读held/官方Validation/Test，不加teacher46/47或states32..35。
每模型每task完整Writer编译一次，共16个新bank，供四state及其J0/JS两格复用；**同模型噪声两格必须是同一个bank文件**。
teacher只输入RGB及exact language，action/state/reward/terminal/ID/路径/pose不进入Writer。
旧dd2采集及bank仅作来源查证；旧128条不能代替本批P_JS，旧P分数不混入主四格。

全部新物化/闭环统一一个clean pushed detached实现E；P训练与RB候选分别保留原7dc/34ea来源，
这是显式冻结权重复用，不要求重做原候选，也不热改旧formal树。pilot后的验收修正按§6唯一阶段例外执行。

## 4. RNG、干预接口与最小验收

canonical其它协议不变：render256/model224、双相机rotate、8D state/7D action、10flow、执行前5后replan、
dummy10、suite horizon220/280/300/520、成功即停、env/policy seed7。
同task/state四格使用相同native-flow seed schedule和环境初态；动作分叉后不要求状态历史相同。

探索新replica固定**4**，不复用旧0..3，也不扫描seed。
精确种子复用纯函数`exploration_seed(global_task,state,4,replan)`，即
`int(SeedSequence([20260925,task,state,4,replan,0x524C]).generate_state(1,dtype=uint64)[0]) & ((1<<63)-1)`。
每decision独立CPU torch.Generator，float32标准normal35乘固定CPU Cholesky转置，time-major reshape5×7，
添加在Source canonical postprocess之前；只改前5步，后45步不改，无额外clip/夹爪override。
P_JS/RB_JS在共同replan index共享探索噪声；J0是严格不加噪声的旁路，不消耗全局RNG。
native-flow与探索子流独立，不为了匹配新旧浮点轨迹改变dtype/kernel或batch。

复用canonical rollout_shard、persistent environment与动态队列，以及现有exploration covariance/纯seed函数。
旧`exploration.py`的screen states32..36且无capture范围不覆盖本批，
旧`return_credit_collection`仅接受replicas0..3并保存信用reservoir，也不适用；不得静默放宽两者旧范围或伪装成采集。
在既有探索所有者添加本合同明确的冻结评测scope，准确校验task/teacher/state/模型/replica/输出根与开关；
其余旧合同保持。调用仍走同一rollout engine，不复活旧RL trainer、不新增并行模拟器或无关framework。

全128条保存compact轨迹、实际动作、每replan policy/exploration seeds及**加噪前归一化5×7均值**；
保留已有加噪后action_chunks及执行前缀。该均值只旁路拷贝现有forward输出，不增加预测；
可CPU复算`u[:5]=mu+epsilon`（J0 epsilon=0）及与反归一化后实际动作的一致，容许正常cast舍入。
逐控制步T+1对象/fixture、EEF、夹爪、全部stage predicates；不增加接触探针。
full固定task2/12/22/34、state0、四格，共16条双相机。

CPU先验收协方差、replica身份、J0旁路、不污染global/policy RNG、scope和均值采集。
唯一真实pilot为task2/state0四格，共4条，计入128；只检查bank/采样/初态/RNG/动作/trace/退出合同，
不按成功排名决定继续。其余124随后完成；无额外工程episode、环境初始化探针或独立10-flow矩阵。
失败保留原件、实际控制步与原因，停止受影响正式执行并报告，不自动重跑。

## 5. 预定读出、停止与资源

五个预定比较：RB_JS−P_JS、RB_J0−P_J0、P_JS−P_J0、RB_JS−RB_J0、交互I。
四个简单差报告R/G/L、churn、Jaccard；四格各报八task、四suite、breadth及全部成功集合。
20000次/seed20260925共同bootstrap：先重采八task，再对每个抽中的task重采四state，四格始终配对；
所有对比复用样本，报告95%描述性区间。交互从同一抽样四格直接作差，不把两个独立CI相减。
每条件只有一次新探索序列，不宣称估计了条件内探索方差；不以显著性选模型。
同state首态函数用现有128个输出：按J0/JS分别比较两模型均值与实际前缀xyz/rotation/gripper及sign，
检查噪声确实按合同注入；后续query分叉不冒称同query误差。
只报告预定成功/动作/谓词与连续几何事实，不将接近/位移自动解释为语义或抓取理解。

无论结果，128完成即主动Queue主讨论原件/缺项/资源并停止新增实验。
本合同不授权改Sigma/步长、补seed、重新估计梯度、连续RL、其它候选或更多teacher/state。
若未识别，不为追逐正结果自动加量；主讨论关闭此有限更新分支或另说明独立信息价值。
即使H_objective得到支持，也须重新论证如何改善J0，不能把打开部署探索直接当EMBER修复。

Study `/data0/user/ymdai/ember_runs/return_objective_alignment_causality_20260925`。
预计.8–1.2 GPU-hours，**硬限1.5 GPU-hours**包含初始化/物化/工程/失败/评测；最多同时两卡、项目≤6。
新data0≤3GiB、开发+formal代码≤768MiB；16bank和128trace保留，复用大资产。
Sol开root前核对strg01独立quota与共享容量，每GPU launch同时核对两节点live身份/所有权/余量。
记录精确命令/env/版本、两个权重及16bank来源、128原始行/退出回执、16case、五对比/函数检查和completion。
计时覆盖整个GPU进程；正常长任务一次持续等待退出事件，不轮询日志/cache。

## 6. pilot共同前缀验收修正的唯一阶段例外

2026-09-25，E=`29634cc9f75143cdd6d70863e6d24ef1a8d3770d`已完成全部16bank与4条登记pilot，
两个阶段全部worker exit0，尚未启动余124。`check_pilot`及分析配对错误要求四格**完整**seed列表相等；
实际replan数20/26/19/20由成功即终止产生，列表长度本来不必相同。
主讨论只读核对所有已执行index的policy及replica4 exploration seed公式、共同前缀、bank、初态、
加噪前均值/实际动作与全T+1 trace，`coordination/main_pilot_pairing_recheck.json`保存证据；
验收不依据四格成功分数，不把此工程错误称为科学阴性。

裁决采用**B：保留E原件，修复正式入口后以唯一新E2完成余124及CPU分析**。
这覆盖§3的单提交要求，仅允许验收/分析及明确阶段来源绑定的改变：

- bank物化来源全部为E，四条task2/state0/teacher34的pilot来源也为E，原路径与内容不改、不重生成或重跑。
- 其它124个事前登记key只允许一个新clean pushed detached E2；E2在实际launch合同中冻结，
  不泛化为任意旧提交均可接受，不改16/4/124/128的数量或来源标签。
- 仅改为非空共同replan前缀配对；每条完整seed列表仍按其实际replan数逐index校验原公式，
  更长轨迹的尾部不能因为“共同前缀”而免检。bank复用、初态、动作注入、trace等验收均保留。
- 可调整`authority`/bank及episode来源验证、pilot和最终分析入口，以分别验证上述阶段；
  policy、Writer、LoRA载荷、探索采样/顺序、rollout、终止、dtype/kernel、数据及科学矩阵不改。
  Sol核对相对E的实际diff并完成针对性CPU验证、真实四pilot只读验收后集成push；不新增GPU smoke或环境探针。
- 最终completion明确记录16bank/4pilot的E、余124的E2及分析实现来源，逐row保留实际commit；
  不把整批写成统一E2。保持canonical evaluator，不新增研究根调度器或monkeypatch旧运行树。

已计GPU用量约.03560779小时，仍计入原1.5小时硬限；最多两卡、3GiB/768MiB及每次launch双节点live准入不变。
修正本身是CPU工作，原有效4条继续计入128；正常退出后统一核验并主动Queue主讨论。
本例外不允许其它行为修正、失败自动重跑或新增评测。
