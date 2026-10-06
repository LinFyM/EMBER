# 共享Writer的去噪路径回报学习：固定终点的有界检验

2026-10-06；依据findings§363。本文登记新的共享训练阶段，不恢复旧Gaussian/RB回报分支。
实际承接、代码身份、资源准入与完成状态只看progress和run root；登记本身不是已启动。

## 1. 问题、依据与限制

目标是让一次编译出的完整LoRA获得更高、可迁移的闭环能力。父T2340为161/400，强MT为153/400；
监督后段没有稳定扩大优势，C及局部功能修正也说明更好获取训练控制不会自动成为held增益。
PB没有支持继续放大冻结内容Reader。这里检验一种不同的学习信用，保持已学视频到参数的实际计算图。

旧9/25回报更新不是未尝试：末端Gaussian探索的一步RAW/NEG均26/64、父35，RB及原采集条件复核也没有净收益。
仅6/32组有非零LOO，两组方向近正交；去掉执行无关夹爪幅度信用没有修复行为损失。见findings§147–150。
本批不以换父点、Adam或多训若干步解释旧失败，也不把稀疏回报、末端Jacobian或FM本身叫作已确定根因。
新主干差别是：探索发生于原生动作生成过程内部，信用直接作用于各次随机转移的velocity；
旧方法只给完整十步生成后的额外动作噪声评分。它可能改变可被回报强化的动作选择及其学习条件，仍可能更差。

这不是findings§151撤回的“条件插值与生成路径不同，所以FM错了”的诊断，也不声称该性质只存在于EMBER。
所要检验的是完整共享G(V,L)能否从自身控制结果取得**正式原采样器下的held增益**；非零梯度、训练收益和SDE收益均不够。
既有负证据降低本假说优先级，因此只给下述单一固定窗口，不扫噪声、LR、seed或追加局部信用探针。

[DPPO §4](https://arxiv.org/html/2409.00588v1#S4)提供把去噪转移嵌入环境MDP、逐转移评分的原理；
[Flow-GRPO §4](https://arxiv.org/html/2505.05470v4#S4)给出rectified flow的反向SDE转换。
它们不证明视频参数编译的迁移，也不保证有限十步Euler与原ODE等价。本批采用下述有限噪声率，不能冒称论文原配置复现。
[FPO §3.3–3.4](https://arxiv.org/html/2507.21053v1#S3.SS3)的FM-loss比率是代理且含ELBO gap；
本批不用它代替真实路径score，也不新增critic或可学习噪声网络。

## 2. 固定父图、数据和信息墙

唯一父Writer为T2340：
`/data1/user/ymdai/ember_runs/operator_read_write_learning_20260928/continuation2340/T/train/attempts/continuation/checkpoints/macro_00002340`。
读其Writer权重，fresh RL optimizer/scheduler；不是SFT exact-resume。沿用T模式的A0/B0及全部P/C/D/O，不换C或加Reader。
原source1000、prefix权重、normalization、processor/tokenizer、38-target rank128/scale1、probe1729的50×32/tau1和双RGB stride5保持。
G的输入仍只有exact language及一条原正确教学视频；每condition在rollout前一次生成完整(A0,B0+M)，四episode共用这一套参数。
source始终冻结；训练梯度通过完整执行LoRA和原native/Writer重放进入全部原Writer参数，包括公共参数经native读取的原有路径。
不detach应有的native信用，不部署第二adapter或推理期Reader，不在任何评测task上学习。

训练任务精确为`configs/operator_read_write_v1/seen_task_scope.json`的原36项（24+12），教学池仍各0..49共50原episode。
采用原T FormalData宏步1..72的task/teacher事件顺序（seed20260928），每宏步四task；只消费task/teacher，不读取原query action作loss。
每36task各访问一次为9宏步，共8遍。视频映射及task权重不因物理分片、长度或回报改变。
这批新增的只是这些训练task上当前policy的on-policy交互，属于AGENTS允许的共享RL标签，不加入正确教学池，不扩大task/demo/外部数据源。
自身RGB/state只进入实际执行policy；奖励、模拟器state、episode身份和动作/去噪轨迹只供训练评分及记录，不能进入deployment G。
validation/test action/reward均无梯度；Test不读。所有缓存、代码、输出新增均在data1。

## 3. 唯一随机生成过程及其信用

每次实际replan仍从官方50×32独立Gaussian初始z开始，执行前5×7，原normalization/限幅/gripper转换保持。
训练和明确标注的SDE诊断采用十步tau_k=1-k/10、h=.1、固定alpha=1：

```
c_k = 1 + .5*(1-tau_k)
m_k = z_k - h*(c_k*v_theta(z_k,tau_k,s) + .5*z_k)
z_(k+1) = m_k + sqrt(h*tau_k)*xi_k,  xi_k ~ N(0,I_50x32)
```

这是sigma(tau)^2=tau的反向SDE Euler离散化，tau=1处有限，最后一次tau=.1也有正方差。
整个50×32路径都按该式随机，不只给前5/前7加噪后把其余theta相关的确定转移当常量；不加中间latent clip。
实际z、m、噪声流和所执行action保持可追溯，普通BF16/TF32及物理batch数值差异按AGENTS接受。
推导使用理想velocity时的score `-(z+(1-tau)*v)/tau`；已学场和粗十步的SDE/ODE分布不同是本批真实风险，不能用连续理论消去。

四episode的初态独立均匀取0..31；seed由20261006、宏步、task、replica及用途分离登记，与worker顺序无关。
每episode的flow初始噪声和各步SDE噪声独立；不按成功挑seed/初态、延时或重采。四episode同teacher，参数整组固定。
R为官方终止成功0/1，无形状奖励、阶段标签、长度奖励或discount；A_i=R_i-mean(other three R)，不除组内标准差。
每episode在Q个真实replan中均匀无放回reservoir M=min(16,Q)，每个保存decision独立均匀无放回选2个去噪步骤。
记录输入z_k、所采z_(k+1)、原m_k/tau、对应真实policy观测和初态/全轨迹来源。采样必须与回报无关。

单步log p为全部50×32元素Gaussian **sum**；唯一actor loss为

```
L = -(1/16) * sum_(four tasks,four episodes) A_i*(Q_i/M_i)
       * sum_(saved decisions) (10/2)*sum_(selected k) log N(z_(k+1);m_theta,k,h*tau_k*I)
```

每宏步只做一次on-policy更新，不重用旧宏步样本优化，不用PPO、trust回滚、参考KL、FM或辅助项。
该式在当前采集参数处的velocity损失余切为 `A_i*c_k/tau_k*(z_(k+1)-m_k)` 乘相应采样/平均权重。
可直接使用保存的m_k形成这一局部余切，经真实单次denoise suffix和完整G的VJP回传；不反传采集路径，也不漏算G。
这给新离散随机policy的回报梯度估计，不是原ODE回报的无偏梯度；未执行维度及不改gripper符号的噪声仍可能增加方差。
不能把log-density元素均值的exp称作真实joint importance ratio；本批根本不计算该ratio。

## 4. 学习窗口、检查点和早停

固定72宏步、1152个训练episode上限；AdamW lr1e-5、betas(.9,.95)、eps1e-8、weight_decay0、clip global norm1。
采用fresh恒定LR scheduler并保存其状态；lr取成熟T监督阶段的既有floor作本次固定小步安排，不据回报调参。
每宏步全部四condition同一omega采集/求导，累积正确等权后更新一次。没有task重加权或把物理rank数当有效batch。
保存每9宏步完整Writer/optimizer/scheduler、sampler/cursor、训练及独立噪声RNG、world topology/schema；63与72为预定读出节点。
资源迁移只在完整checkpoint边界并守逻辑流，不称跨拓扑bitwise exact。原SFT optimizer不恢复。

若下面的父SDE144全部无成功，直接停止，不进入训练；若首9宏步的全部36组LOO均为零，停止无信用分支，不扩大探索/采集。
这两种停止都不把相同父函数重跑400。其余正常科学分数不改变72终点；nonfinite、信息墙/配对失效或预算风险按工程/科学边界处置。
63一旦完成可与余下固定训练并行物化/评测，结果不得反哺剩余训练、lr或样本。任何科学阴性不作为bug修补。

## 5. 固定读出与完整判断

训练前在原36task×init32..35的seen144面板完成父SDE144；使用原video_schedule seed20260928及原scene/env/policy初始噪声映射。
父ODE使用已有T2340的29/34/73三task共12原行（query_conditioned_transition_read_20261005），只补其余132行；不拿T1800代替。
父SDE与父ODE仅用于标识采样分布代价，不按其分数改alpha或优化器。
末点72在相同seen144分别做原ODE和SDE，各144；两个共享模型节点63/72各做严格原ODE correct400。
共最多1364条新评测行、1152训练episode；原T161/400和强MT153/400直接复用，所有旧行保存准确来源。
400为validation [3,6,11,16,23,26,31,39]×0..49，50条合法视频整轮各一次，沿原state-video/scene/env/policy RNG配对。
official保持render256/model224、双相机rotate180、8 state/7 action、**原十步ODE**、前5 replan、dummy10及suite horizon，成功即止。

末点72是唯一方法候选，63只提供相邻证据；不选两点较高者、联合成功或根据SDE回报选checkpoint。
每点报告task/suite、breadth、对父/强MT的R/G/L/churn/Jaccard，两个相邻success-set及全部不利任务。
train/validation分开，SDE/official ODE分开；不把seen144当held，不用它改checkpoint。固定144面板重复视频的范围如实登记。
完整raw rows、真实动作/连续状态/原生成功、每条件LoRA来源、训练事件与采样权重、group回报/非零率、各原参数组grad norm均保留。
validation各task init0 full；seen144各task init32 full，其余compact，沿现有capture保存真实RGB/动作/状态，不能把最终未拍帧补成见过。
需要报告的行为变化从这些固定画面与原记录读取；不添加姿态/接触oracle、额外环境续行或梯度探针。

- 只有SDE自身成功提高而原ODE不提高：学习了另一执行分布，不能算目标收益，不调noise补救。
- seen ODE提高但两held点无有覆盖净收益：强化了训练控制，未证明视频编译的迁移；不因train高点续训。
- held只单点小涨或伴明显相邻能力丢失：保留完整得失，不把偶然高点作为稳定方法；停止本窗口。
- 两相邻official点都形成相对强父/MT有覆盖的实质收益：才提高该共享学习原则的支持度；仍需main另行判断视频有益增量及后续资格。
全部分支完成即停止、整批回main；没有自动延长、第二alpha/LR/seed、critic、FM混合、Task-local RL、video controls或Test。
负结果约束本次完整去噪回报学习假说，不等于所有RL或全部固定数据内方法不可能。

## 6. 执行、资源和生命周期

唯一root `/data1/user/ymdai/ember_runs/denoising_return_writer_20261006/`。
预计含工程4–6小时；硬承接起8wall小时、20完整GPUh、48GiB新增峰值，所有加载/profile/失败/恢复/临时/冻结树计入。
按旧四rank T约15–17秒/112-query宏步，最多512个单去噪评分/本宏步估计训练约5–7GPUh；
2516个训练/评测episode按既有persistent实测粗估5–7GPUh，另留实现/物化余量。不是本批已经实测的吞吐。
bank约25–32GiB、完整恢复/训练采样证据约6–10GiB，流式临时≤4GiB；先实际测量父checkpoint/bank布局再登记峰值，复用大资产。
超过6h预期或累计15GPUh仍不能完成剩余固定面板时回报具体预测，硬限不得静默越过；不得减面板、换精度或重复已完成行凑结果。

实验session负责新独占codex worktree、实现、实际消费者检查、Git集成及clean pushed detached运行；main不重复工程验收。
首launch及resume前核strg01 data1独立quota/个人实占/共享容量，同时live核两GPU节点及当前累计占用，遵守现行8/6/单节点6上限。
按实际吞吐选择训练rank和可并行的物化/评测，不额外从四逻辑task推导整批物理卡上限，不跨节点拼训练。
复用原native编译/LoRA functional/VJP、资产及官方persistent dynamic evaluator；新增仅本共享RL阶段所需采集、score及训练owner，不复活旧大trainer。
应用结构检查并登记新模块职责/行数；旧`pi05_eval.return_credit`是历史seed兼容面，不把它静默改成新RL owner。
在本批原事件上验证真实转移/score余切/完整38 LoRA信用、源权重冻结、信息墙及采样/任务权重；普通数值容差，不做逐tensor/逐bit追查。
显存有余量主动测更大评分microbatch、frame chunk和persistent replicas；最多两次丢弃profile更新后恢复父/optimizer/RNG。
profile只复用初始父参数下宏步1已采集的合法训练轨迹（init0..31），不能用seen诊断的32..35作丢弃更新；
恢复后该宏步仍可按原采集参数使用一次。不额外生成环境科学case，所有profile计费并保留吞吐/峰值和未再放大的依据。

启动后直接等真实退出事件，不轮询训练分数/缓存、不心跳或逐阶段自通知。整批完成或科学/预算实质阻碍只回main一次。
保留run contract、源码身份、训练来源、checkpoint/原始采样与原行/aggregate/completion及完整资源账；图表可从原件生成，不补算新模型/环境。
本批结束封禁专用run的重启入口；候选未获得上述完整收益时退役专用执行面，以Git/frozen/原件保存，不留下第二条常驻RL路径。
实验session清理已核实结束的自有工程worktree并push，交回canonical tracked/Git窗口；没有自动后继。
