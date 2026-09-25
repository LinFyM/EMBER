# 成功信用能否形成有用的共享更新方向

2026-09-25。是否active只看progress；机器合同`configs/return_credit_direction_v1/experiment_spec.json`。
主讨论01a0cd94-65da-7b22-8ca9-7ba35f454632；执行者为现有Sol 01a0cd90-ebb7-77a1-a20b-a858825d2f66。
这是一批共享Writer的有限方向干预，不是长程RL、部署时适应或新的模型选择。

## 1. 证据、问题与边界

§146的TASK没有优于BASE；不继续lookahead训练，也不因正常BF16求和余项追加精度实验。
§135中C的实际演示前缀MSE小于B，闭环却更弱；§145中plate方向与最终成功也不单调。
§127的任务隔离、§129的缩head及§130–131的大task batch没有构成修复。与此同时，private Writer局部学习、
旧有用P贡献和Source→Object获取仍是正证据，不能断言共享图无学习能力或视频没有信息。

本批检验：同一个现有Writer，来自自己闭环成功的梯度，能否产生比普通FM信用更有用、且可跨正确视频复用的方向？
若成功信用方向在独立条件增益、反方向损害，而FM没有相同收益，则支持这一起点存在可利用的行为信用；
它仍不单独区分目标函数与expert/on-policy状态分布，也不证明FM普遍错误或已找到统一根因。
若梯度仅拟合采集条件、收益不跨video/state，或无足够奖励变化/函数作用，则本窗口没有识别出可迁移信用。

四臂C的420/630/840/1050/1260为110/105/101/110/100，完整有限监督没有稳定上行；关系支持也未修复。
这些事实和上述机制诊断允许一个有界信用试验，不据“平台”直接恢复旧RL大训练。
仍固定已用于§145–146的C_S00@1155，不按新奖励或held分数再挑起点。
9月8日e1ea3596的完整10-flow VJP与Gaussian profile只有工程证据；旧SEOD/GOMQ是expert occupancy蒸馏，
本批使用当前policy自己的实际成功回报。区别是真实信用来源，不是把endpoint或expert标签再命名。

## 2. 两个数学对象，不混称同一风险

令一次编译得到的policy动作均值为`mu_phi(q,xi)=first5(F10(theta0+G_phi(V,L),q,xi))`，35维。
采集时在规范化动作空间取`u=mu_phi+eta`，`eta~N(0,Sigma)`，再走原postprocess/controller；不额外裁剪。
xi为原生flow噪声。Sigma沿用已明确的物理尺度：时间相关`C_ij=.8^|i-j|`，
`Sigma=C⊗diag([.05^2]*6+[.10^2])`。这只是本批固定探索分布，不宣称最优或恢复旧trust协议。

对固定teacher/task/init-state的四条独立探索轨迹，`A_i=R_i-(sum_{j!=i} R_j)/3`，R只用官方binary success。
在每条轨迹Q个真实replan中，以独立RNG均匀无放回保存`M=min(4,Q)`个decision；其贡献乘Q/M。
32组、每组4条，最大化回报的估计方向为

`g_R = (1/128) sum_groups sum_i A_i * (Q_i/M_i) * sum_saved J_mu(q,xi)^T Sigma^-1 (u-mu_old)`。

环境状态、实际采样动作u、xi、A及采集均值mu_old全stop-gradient；通过真实十次Euler和完整A/B得到cotangent，
再完整重放Writer，三Meta与其它可学习参数全部活动。不能只通过最后一次denoise或action_out。
若最后一块提前成功，采用已实际采样的完整35维latent u及其联合密度；未执行尾部是积分掉的辅助随机变量，
保留它仍是该采样policy的无偏score估计，可能增加方差。记录实际执行mask，不伪称执行了尾部。
成功之后不再采样新decision。dummy settling不计policy decision。

对同八task、同八teacher，另复用原S00 occurrence165的21+7跨episode随机tau/fullH事件：
`L_F=(1/8)sum_task(mean21 FM + (1/3)mean7 FM)`，`g_F=-grad L_F`。
原224query及noise不重抽，梯度仅用episodes0..45。FM使用expert状态，R使用on-policy状态；这是明确保留的竞争解释，
本批不能把差额全部归给某个loss公式。

每个方向只走一次fresh SGD，无momentum、weight decay、clip、继承Adam或scheduler历史：
`phi_R=phi+r*g_R/||g_R||`，`phi_NEG=phi-r*g_R/||g_R||`，`phi_F=phi+r*g_F/||g_F||`。
r固定为上一批七个BASE实际参数位移的RMS：**0.17076466164261486**。这只是一个已实际使用的参数步幅，
不保证相同功能步幅、相同KL或相同预期收益。保存真实有限动作变化，不按结果调r或扫描探索尺度。
同一个g_R的正负方向用于方向性对照；FM参照检验是否只是任何当前学习方向都能产生相似收益。
三个是独立父状态的一步结果，不连续应用、不组合、不选择；P为未更新父。

在可微且有限步局部近似成立时，对真实回报目标有`delta J = r * <grad J,d> + O(r^2)`，d为单位更新方向。
若估计g_R确实对准可泛化的真实梯度，反转它应反转一阶作用；FM方向的符号由它与回报梯度的关系决定。
有限样本、非线性以及新video/state迁移都可能使该预测失败，不能用公式代替验证。
尤其g_R对应有探索的J_Sigma，正式独立面板测无探索的J0，两者不相等；负结果还保留探索分布不匹配，
本批不自动补J_Sigma评测或声称已否定所有回报信用。FM格是等参数步幅的FM梯度方向，不冒称原Adam训练更新。

## 3. 固定数据与一次采集

八个合法fit28 task：`[2,5,12,17,22,25,34,37]`，每suite两task；它们均非诊断held8/官方Val/Test。
原事件index/teacher为`4620/34,4622/13,4624/38,4626/8,4628/5,4629/36,4633/2,4635/45`，
均为occurrence165。固定语言、Source1000、normalization、agentview教学/stride5/真实末帧与父模型全部保持。
每task固定这一条teacher，init states0..3各四探索replica，共128条；不是50-video正式评测或性能数字。
组内同初态与native flow RNG，探索RNG按replica独立。每个decision独立CPU Generator，不能依赖worker/batch顺序。
同组不同replica的flow噪声公共前缀相同，但执行观测随动作演化；LOO在独立探索噪声下成立。
四个初态的native-flow公共子流与replica探索子流的种子定义以机器合同为准，不能静默借旧exploration入口的旧init-state白名单。

第一组task2/state0四条作为正式pilot，计入128。先验收真实采集→相同LoRA的10-flow重放/VJP、信息墙及score数学。
即便该组回报相同，也继续其余登记组；不能按pilot成功率换task或探索尺度。
最多4条独立工程smoke只许合法task2，不作formal初始化或正式分数；若无需新episode就不消耗它们。

采集后所有32组及每task回报完整公布；若全部A=0，则g_R无可辨识信用，停止候选与后续评测，保存负结果。
非零但不可靠的梯度仍按固定合同计算，不靠追加replica追显著。按init-state偶/奇分别汇总两个独立组集合的方向，
每半组均覆盖八task；只作SNR/状态依赖诊断，不按其cosine选择更新、task或权重。
新环境成功标签只用于以上训练task的共享g_R，不成为Writer条件。不读取held inputs、held expert或官方Val/Test。

## 4. 独立闭环：最多256条，完成即停

P/R/NEG/FM四格，每格八task×states32..35×teacher46/47，共64；总256。
两条teacher均为同task合法正确视频，分别单独编译一套完整LoRA；不融合、不取优。
这是一份明确允许固定teacher复用的机制面板，不称50-video无放回、strict400或模型资格。
不加探索，执行完全canonical：双相机、8D state、10 flow、前5步replan、dummy10、成功即停、原suite horizon。
四格同task/state/video的环境与原生policy RNG配对；teacher46/47也复用该状态的同一RNG，不把不同teacher混为独立state。
eval seed7；全部bank与结果来自同一新clean pushed detached提交，P也在本批重新评，旧分数不拼入。

固定六对：R−P、NEG−P、FM−P、R−NEG、R−FM、NEG−FM；每teacher单列以及两teacher等权汇总。
每task/suite完整成功表、R/G/L/churn/Jaccard、两个teacher的成功集合与对比全部保留。
bootstrap20000/seed20260925：同一组抽样共同用于所有格/对比，先重采样八task，再在各task重采样四state，
teacher两格保持成对。少task、单训练起点/采集seed，仅描述性区间；不据六个对比中最好一个采纳方法。
没有相邻学习节点、未见task泛化或相对语言baseline的证据，不能称有益视频或稳定修复。

全128采集与256评测保存动作/噪声/8D states、T+1连续object/EEF/gripper及stage predicates。
工程必须先审计这八task的对象/fixture/arena region采集，复用已修复passive机制，不能跑完才发现缺项。
允许每task一次、共最多8次无动作环境初始化/身份探针；无env.step/policy forward，不作正式episode或科学分数。
训练额外保存最多512个真实decision的模型输入RGB/state、flow noise、原均值、未裁剪u、batch provenance、Q/M与executed mask，
支持完整score/VJP复核；不能用改变了当前policy的新rollout补齐缺失输入。
三个候选在这些已保存decision各重放一次，最多1536次10-flow；父的VJP/核验最多512个独立decision输入。
保留full50×7输出及前5环境/OSC位移，报告R/NEG/FM实际功能步幅；内部checkpoint重算不伪算新独立query。
固定full cases仅评测task2/12/22/34、state32、teacher46、四格，共16条双相机；不展开其它全量图像。

## 5. 预先裁决与工程边界

H_credit：R在独立state和两个teacher上相对P/FM取得同向实际成功收益，NEG反向或明显弱；
支持当前共享图存在可利用的闭环学习信用。还须保留能力交换及收集分布限制，不能直接归罪FM或宣布视频必要。
H_fit：只有采集侧方向/score改善，独立闭环没有对应收益，或只对一条teacher成立；不继续大RL。
H_any_update：FM相当或更强，或R/NEG都近似；本批没有分开成功信用与普通学习/扰动作用。
H_unidentified：全组无reward variation，函数位移太小，或梯度估计状态依赖/方差过大；只记录本预算未识别，不放大方向。
H_exploration：所学方向针对J_Sigma而没有迁移到J0；本批阴性不能与这一解释分离，必要补测须另说明信息价值。
本批不预授权多轮RL、更多采集/评测、重新选父、trust回滚、小扫或架构更换；主讨论核验后另裁决。

复用canonical episode/evaluator、LoRA物化与Writer VJP；可从e1ea3596读取已验证10-flow梯度原理，
不能恢复旧joint trainer、旧`.1/16`信用系数、旧task白名单或trust halving，当前系数由本合同给定。
不要另造平行policy或完整rollout引擎；保留一个当前可维护的信用/flow owner与薄诊断driver。
先验证采样协方差、LOO、Q/M、梯度符号/归一、三Meta活动、Source冻结和候选独立父状态。
flow重放保持原生mixed dtype及真实prefix；记录原均值vs重放均值RMS/max。
正常BF16差异不要求逐bit消除；若前5规范化动作RMS超过.01或实际破坏score/VJP语义，暂停查明具体错误，不改dtype追数值。
训练/物化时不跨参数状态缓存适配激活。父完整checkpoint只读；新三格完整checkpoint标明新SGD step1、
fresh optimizer/scheduler/no sampler continuation，同时保留gradient/cotangent及必要分组证据，不能冒称父Adam exact-resume。

Sol从最新main隔离开发，必要的结构检查与直接接口验证后集成推送，再冻唯一提交。采集/梯度/新bank/评测同一提交。
每阶段原子写completion，恢复只补未完成组或面板；partial轨迹/失败工程原件保留，不拼接成有效episode。
study `/data0/user/ymdai/ember_runs/return_credit_direction_causality_20260925`，新data0≤8GiB，代码≤768MiB，
正式计算预计2–3 GPU-hours，硬上限4 GPU-hours；pilot后用实际吞吐确认剩余预算一次，不按奖励判断是否扩量。
梯度优先同节点world2，独立rollout/物化按cost-balanced queue用真实有益空卡，项目合计≤6物理GPU。
新root前strg01独立quota/共享容量/峰值检查；每launch同时live检查gpu01/gpu02。正常任务一次持续等待退出，不轮询日志/cache。
完成或真实阻塞后主动Queue主讨论原件、缺项、实际资源和结果，停止新增实验。
