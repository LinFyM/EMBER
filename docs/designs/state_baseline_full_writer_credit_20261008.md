# 状态回报基线是否改变完整视频编译梯度：固定历史原件辨识

2026-10-08；依据 findings§398–399，经 progress 登记后为唯一 active 分析。
只在六个与原采集精确对应的旧 checkpoint 上重算三种基线的真实梯度；0 参数更新、0 新环境。
这不是 RL 恢复、checkpoint 选择或完整方法资格。上一 CPU 批保持封口，预测器不重新拟合。

## 1. 完整问题、现有证据和本次决策价值

EMBER仍未稳定明显超过强MT；正确视频写出的控制还未可靠跨自身状态迁移。
最近困难教师480从初态12/16、T50接手9/16，task29接手0/4；task12同权重初态搬对对象、接手却搬错。
旧完整T的72步SDE回报学习确已更新全部G/native路径，seen SDE109→109、ODE109→102、official161→154/158。
不能再次把接通score或“采用RL”当修复，也不能假设已有可靠恢复教师可供蒸馏。

旧LOO把一条episode的优势广播到全部抽中时点。CPU三前向窗显示：Context已解释较大task/时间差额，
State额外使Brier降低14.6%、局部velocity余切能量降低15.4%，后两窗有较强预测增量。
相对LOO，State的最早时间四分位局部能量比.9988，中后段为.7163/.2957/.1273。
这支持“可能少把低成功概率状态上的随机动作广播成更新”的候选；不支持已找出早期正确操作方向。
全失败组State反而增加原LOO为零的信用，task25/38的正知识仍不足；这些反例不能由总体降幅掩盖。

尚未区分的解释是：局部变化通过执行LoRA和视频生成Jacobian后，仍能减少有限样本的完整更新二阶量；
或者Jacobian及跨时点/跨condition叠加抵消了收益，甚至只改变公共参数、使视频读写信用更不稳定。
后者成立会降低“追加状态基线即可改善完整G学习”的优先级。前者才支持考虑一次真正的有界学习比较，
仍须有闭环效果，不自动授权。只做这一轮真实G桥接，不按结果追加预测器、正则、参数组截断或局部扫描。

## 2. 实际计算链与可辨识的数学量

每个原condition使用采集时的 `Theta_c=G_phi(V_c,L_c)`，教学仍只有合法视频和exact language。
自身原观测 `o_it`、抽中 `z_itk` 和 `tau_k` 进入冻结source及完整38-target执行LoRA，计算真实velocity。
原 `z_next,m_old` 均 stop-gradient；不对已采样整条SDE、环境状态或预测器反传。
三种基线只改变同一实际velocity输出上的损失余切：

```
b in {原LOO, 冻结Context, 冻结State}
c_k = 1 + .5*(1-tau_k)
q_itk(b) = (R_i-b_it)/16 * (Q_i/M_i) * 5 * c_k/tau_k * (z_next-m_old)
dTheta_c(b) = sum_itk (D_Theta velocity_itk)^T q_itk(b)
g_c(b) = (D_phi G_phi(V_c,L_c))^T dTheta_c(b)
g_macro(b) = sum_c g_c(b)                 # 原4条件、16 episode有效权重
```

恢复CPU批约掉的公共1/16；不做按时长、frame、优势标准差或任务重新归一化，不裁剪，不经过Adam。
必须重放**完整同版本G**：真实图文prefix→public A/B0作用下的native X/H→P/C/D/O读写→完整A/B。
T的A是公共A，B=B0+M；H的RMS相邻差、`P(A X)+C(H)`门、`D(delta H)`值、O及delta-rule写入沿原源计算。
公共A/B0还影响native读取，不能detach视频特征或只给最终公共LoRA分支信用。source基础权重始终冻结。
参数组是计算坐标，不是知识互斥分区；单列P/C/D/O只为了看变化是否到达实际视频读写算子。

令F包含该macro之前的训练历史、固定任务/教学/初态安排和已冻结预测器。State/Context只看当前动作之前的信息，
原LOO只看其它独立episode的R，所以每条score上的基线项条件期望为零。Q/M是原均匀reservoir的校正，
对抽样取期望后恢复完整已存活时点之和，不把依赖终止的Q当预测特征。因此三估计器有同一条件期望mu：

```
E[g_Context | F] = E[g_State | F] = E[g_LOO | F] = mu
E[||g_Context||^2 - ||g_State||^2 | F]
  = tr Cov(g_Context | F) - tr Cov(g_State | F)
```

这一恒等式说明成对平方范数差为何有判断价值，**不证明六份实际样本已经降低真实方差**。
原LOO共享其它三条回报，四条episode梯度非独立，不能套独立样本方差/SNR公式。
六个phi/condition组合不同且训练历史自适应，不能把其方向当一个固定目标的重复梯度平均或报独立重复置信区间。
实际比值只称“成对梯度二阶量比”，不是方差比；小范数、cosine或非零native信用均不等于正确操作或性能收益。

## 3. 唯一cohort、checkpoint和预测器

旧root `/data1/user/ymdai/ember_runs/denoising_return_writer_20261006`，采集/score源 `1c90e7d5681bf18c99f9dc4084225d03e76ee929`。
选择规则：在三个已冻结前向预测窗中，取所有紧随已保留完整9步checkpoint的原macro；不依据R或预测降幅筛选。
不能拿别的checkpoint给原轨迹评分，也不能重新采样以补齐缺失的参数版本。

| 原采集macro | 只读checkpoint macro | 冻结预测器拟合范围 | 原task | decisions |
| --- | --- | --- | --- | ---: |
| 19 | 18 | 1–18 | 29,35,56,64 | 256 |
| 28 | 27 | 1–18 | 5,12,62,97 | 256 |
| 37 | 36 | 1–36 | 7,36,55,97 | 256 |
| 46 | 45 | 1–36 | 15,19,20,51 | 246 |
| 55 | 54 | 1–54 | 0,2,34,38 | 255 |
| 64 | 63 | 1–54 | 4,17,32,55 | 256 |

总24 condition、96 episode、1525原reservoir decision、3050原选中转移，22个不同task。
12 mixed组、12全成功组、无全失败组；76/96成功是这份历史cohort的事实，不是新能力分数。
task25不在cohort，完整CPU最大三个净收益task34/37/73中只有34在；不能声称代表全部36或解决全失败组的问题。
选择后旧局部State/Context比分别.9857/.9356/.9031/.6019/.8780/.8580，作为实际配对参照保留，不用于删点。

只读取原checkpoint的Writer权重、原group/event/rows、原score与同一teacher视频；不恢复optimizer、不执行原训练入口。
仅从 `state_conditioned_return_analysis_20261008/analysis/per_decision_predictions.jsonl` 取这1525个已存预测，
以macro/task/replica/replan/t/selected steps精确对应原事件；它们已经由先于该macro的模型产生，不重新fit或clip。
保留原8 state、双相机处理、50×32 latent全部维度、两个原tau、Q/M和四episode全部R；同一decision的两转移共用同一b。
全成功组原LOO为零，其梯度确为零；Context/State必须真实计算，不沿原LOO-zero捷径跳过这12组。
禁止held/Test、时序controls、未来标签、教师特权条件、新action query、新环境或任何参数更新。

## 4. 固定输出、反例和判断分支

每个condition、每macro输出三基线的完整梯度3×3 Gram、norm、配对差范数/cosine；
同时按public A、public B0、P、C、D、O及联合P/C/D/O分组。零梯度的cosine/比值标空，不加epsilon伪造可比方向。
独立S不存在，保留“非活动”身份，不建立新参数支路。报告完整228个trainable参数覆盖、source冻结与finite事实即可。
同样记录完整38-target LoRA余切的Gram/norm，作为local velocity→LoRA→完整G的实际边界，不新增loss或学习门槛。
对每基线报告 `||g_macro||²-sum_c ||g_c||²`，保留跨condition交叉项；不能把各组平方范数之和当真实macro更新。
保存三种macro梯度FP32原数组及name/shape顺序，共18份约2.84GiB；condition只留完整统计，避免保存重复大中间量。

并列原cohort的Brier/E_local、6 macro和24 condition的全部数据、mixed/全成功分层、12/29/32/38原例，
以及合并平方量比、成对差均值/中位数和正负点数。保留每个State变差、原LOO为零而新出现信用的组。
先在每个同phi/同事件内比较，再汇总标量；不平均六个不同phi的梯度来声称“真实方向”。
原LOO macro范数可与旧metrics数量级核对，普通dtype、物理batch和reduction差接受，不建立逐元素/逐bit复现门槛。

若State相对Context的局部增益经过完整G仍有一致、实质的减少，且P/C/D/O没有相反恶化，
可提高一次状态基线实际学习比较的优先级；仍不能由六点降低直接启动、选择新模型或宣称闭环修复。
若只有Context有效，收益先归task/时间；若完整G/视频组消失或反转，不追加critic扫描保护状态假说。
若差异集中个别macro/公共B0或被全成功组新增信用抵消，明确限定作用；不通过换分组、归一化或删反例追回阳性。
只有这份历史条件下的证据，不能证明所有actor-critic无效，也不能替代后续实际能力/迁移判断。
main完成原件消费后自主形成下一完整方法决策；本批没有自动RL/蒸馏/扩专家/新Writer或更大矩阵。

## 5. 执行、资源和交付

唯一root `/data1/user/ymdai/ember_runs/state_baseline_full_writer_credit_20261008`。
预计工程至完整交付1–2wall小时、总.5–1.5 GPUh；原实测每128转移score约9.4秒、完整native重放约6–16秒，
24组×3基线约72次重放，已无占旧RL主要时间的环境采集。加载、profile、失败和读出仍完整计费。
硬上限从实际承接起4wall小时、3完整GPUh、data1新增峰8GiB；超期/缺原件/科学改变回main，不缩cohort交阳性。
启动前核strg01 data1独立quota、相关个人实占/shared与梯度/冻结代码/临时峰；原模型和captures只读复用不复制。
每次GPU launch同时live核两节点，按AGENTS当时全项目8/6及单节点6上限；无额外短分析例外或dummy占卡。
六checkpoint任务可独立并行或让持久worker顺序换权重，按实测吞吐和现场容量安排；原world3不是本批上限。
无需DDP/NCCL或迁移optimizer；同macro四condition可流式求和，三个基线可复用同一次velocity/native forward的多VJP，
以实际图/内存正确性决定物理安排，不能detach任一路径来省时间。

对本cohort的真实组做少量物理profile，验证更大microbatch（原32起，可64/128）及有余量时更大frame chunk（原16起）。
固定事件/余切/有效权重不变，无optimizer更新；把吞吐、显存峰与未继续放大的实测依据保存，profile成本纳入硬上限。
记录实际消费者的事件/checkpoint/预测器时序、完整LoRA/native信用及finite，不加通用测试框架、hash、逐tensor一致性扫描。
复用旧封存源码或现有canonical owner；新增诊断代码须有唯一内聚owner，从新clean pushed detached版本运行，旧冻结树不热改。
不恢复旧RL训练面；仅为本批需要的临时入口封口后退役，脚本/原件/代码身份保留。工程实现/消费者验证/Git由实验session闭环。
main交接后停止tracked/Git并发写；运行使用实际退出事件等待，不轮询日志，不逐阶段/selfQueue回报。
完整结果写canonical分析JSON、findings/progress/task_plan/history，保留18梯度、全部统计、原件索引、命令/成本/失败/完成记录。
整批或真实合同边界一次可靠回main并交回写窗口；main继续Owner目标，不把这一辨识结束当作项目结束。
