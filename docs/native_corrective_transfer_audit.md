# 原生条件与真实纠正的跨episode传递诊断

2026-09-13。本文登记一个训练侧privileged诊断，不是新的action-hidden Writer或部署方法。
它检验一次由真实动作误差构造的完整LoRA能否改变另一episode的实际策略输出。
旧Writer及物理效果诊断均保持关闭；当前执行状态只看progress。

## 1. 问题与最近历史

双相机source的原生动作读出有正证据；它不证明视频预测误差可以转换成执行者另一状态下的正确更新。
本项先给纠正信号真实动作标签，固定参数构造算子，移除新读取器与共享Compiler的拟合混淆。
这比再训练一个局部动作头更直接：即使给定正确纠正，若该算子仍不能传递，就停止该具体出口的投入。

已有G1/P1局部容量、J2跨episode真实FM正控及EBSRI/PNBTT负例必须保留。本项不声称首次使用行为梯度：

- 旧95-task behavior authority保存八target的rank4梯度因子；源码
  `fcdb6e43:src/ember/ecp/behavior/gate.py::_exact_rows`中的.716/.801、.2695等恢复量是`factor_cosine`。
  `5cbe76e0:scripts/seal_ecp_g2_behavior_codes.py`只加载外部因子，明确`source_policy_loaded:false`。
  本次有限审计未定位初始梯度生成器，不能推断其source/carrier运行点，也不能把几何恢复写成安装更新后的FM收益。
- P1 `c9e8198`以expert更新consensus初始化free primal，fit/held .9717/.9545是mapping/subspace容量。
  J2 `f3677a5`的task-local正控确实用真实FM优化100次，10/10 Panel-B优于carrier；共享`5fd80b6`仍不足。
- EBSRI `25477c9`与PNBTT `e65c6388`将LoRA叶VJP作为共享生成器的训练信用；forward中的B仍从native Y signed span产生。
  PNBTT E1优化free queries，真实Program E2未启动。它们不是教学同帧输入与输出cotangent的直接外积构造。

本项的限定区别是：裸冻结source、实际同帧动作纠正、全部38目标的原始weight导数、一次rank16构造，
随后直接计算另一episode上的原生输出。没有carrier、expert因子目标、free task code、共享映射学习或Y输出span。
它仍只是具体算子的传递前提；不能由privileged通过替RGB获取、任务迁移或闭环背书。

## 2. 真实功能联系

令z包含真实双相机观测、exact language、登记的state输入合同、noise及flow time。
冻结source在t=1输出完整velocity F0(z)，真实前15步动作a给出目标u=ε−a。
对某个目标线性层y_l=W_l x_l，有

```text
r_i = F0(z_i)[:15,:7] − (ε_i − a_i)
L_T = mean_i,h,c r_i[h,c]²
g_l = ∂L_T/∂W_l = Σ_i,h c_lih x_lihᵀ
D_l = −rank16(g_l)
F(W0 + ηD, z′) = F0(z′) + η J(z′)D + O(η²)
```

c是同一次原生forward在该层输出上的真实cotangent，包含下游动作误差的链式传播；它不是Y或两次forecast之差。
全部50个H在真实网络内参与计算，只在最终监督出口评分已观察区间对应的15×7，未提前平均／截断native H。
原生输入x与对应cotangent配对；B因此不被限制在原先的Y列空间。rank16仍是当前诊断的有限容量。

展开跨状态作用可写成J(z′)J(z)ᵀr，低秩压缩会进一步改变它。教师位置上的下降不保证z′上的下降，
state-free teacher与含真实state的执行输入也不具有自动相同的Jacobian。**这正是本项要实际检验的联系。**
其中没有“几何相似就证明动作收益”或“辅助头正确就证明Compiler有效”的跳步。

在teacher样本上计算精确方向导数d=J_T D，固定一次构造幅度：

```text
η = 0.5 × max(0, −〈r,d〉 / 〈d,d〉).
```

这是teacher线性化平方损失最优幅度的一半；不看独立query标签定幅，不试多个η，不线搜索、回滚或追加更新。
若方向或分母为零，生成identity并记录原因；若实际非线性输出更差，照实保留，不改幅度挽救。
各层梯度采用固定q=24、niter=2的随机低秩SVD（小维度目标取可用全秩），固定seed20260923；最终截取16维。
只保存一套rank16 A/B，physical source参数始终冻结且未写入；没有optimizer/scheduler。

## 3. 固定数据与两个oracle臂

- 固定train24，不增加其它meta tasks，不读取validation/test动作或outcomes。
- 构造池仅demo16–19。每episode取四个等分附近的合法起点：将有效stride5起点序列的1/5、2/5、3/5、4/5索引向下取整。
  每个起点p对应已观察四帧p/p+5/p+10/p+15与actions[p+1:p+16]；本项oracle直接使用真动作，尚不运行视频反演。
  每条episode的四位置等权形成一套LoRA，不平均各片段／各video的最终LoRA。共96条件，每臂96套。
- 传递query复用既有Local诊断的demo42–45、每episode四位置，共384个固定位置；所有动作只在预测后评分，无query梯度。
  每task的四套LoRA分别用于该task全部16位置，共1,536个condition-query组合／臂。
- `state_free`：teacher原生prefix只接双相机RGB与exact language，无teacher state。
- `state_true_oracle`：仅作为privileged上界参照，在teacher prefix加入实际8维state；动作标签、位置、noise和构造完全配对。
- 所有传递query都使用执行者自己的真实观测与8维state；它是policy合法执行输入，不属于teacher信息墙。
  两臂分别构造，不能合并为一个含真state的部署Writer。

teacher固定public Gaussian probe seed1729、t=1。query每位置使用seed `20260924 + 100*task + clip` 的独立完整50×32 Gaussian，
两oracle、四teacher与source严格复用。同一初始noise分别计算t=1端点估计和官方10步Euler生成，均评分前15×7。
两读出共同登记，不在它们中选优、不改变正式10步rollout合同。不做环境step或任何完整闭环。

## 4. 报告与停止条件

保留每套LoRA的构造样本、teacher loss前后、η、完整38目标、原始预测、逐query MSE与两读出；
source参照在相同query/RNG合同下新算一次并复用。先对query与四teacher等权汇总task，再对24task等权。
报告全部per-task/per-suite及跨teacher结果，不筛选成功任务／视频。20,000次task-cluster paired bootstrap，seed20260925。

一个oracle的具体传递前提仅在**t=1与full10两读出**的source−oracle净改善95%CI下界均严格>0，
且两读出各至少两个suite净正时通过。它没有145/400、selected checkpoint或正式视频资格含义。
完整非线性策略输出是裁决依据；低梯度MSE、低秩能量、teacher拟合、线性预测与norm仅作定位。
若某个完整oracle产生非有限预测，保留原始数组并把该臂记为non-pass，不能删除该条件后重新计算通过率。

| 结果 | 改变的投入决定 |
| --- | --- |
| state-free通过 | 固定原生条件＋真实纠正的这个出口具有跨episode功能前提；才继续推导合法RGB纠正获取及无task-local优化的摊销生成。不自动启动完整Writer |
| 仅true-state通过 | 当前state-free条件与执行输入之间的传递未成立；不把失败归到视频反演精度，也不默认重开已关闭的state预测／Meta校准扫描 |
| 两臂都不通过 | 停止这套t1、一次rank16纠正构造作为后继出口；不训练以它为前提的RGB残差头，不追加η/rank/probe/flow或轮数扫描 |
| 端点改善但full10不改善 | 端点方向不能据此当作完整生成的纠正，按上述两读出共同资格non-pass；不把部署改成一步采样 |

没有ordered/frame_set比较、wrong或shuffled/reversed controls；本项不调整其现行资格，也不借oracle结论选正式方法。
Source在teacher轨迹上误差为零而新初态失败的反例仍成立；本项不是“动作残差包含全部视频价值”的充分性证明。

## 5. 实施、资源与部署边界

实现使用现有source loader、native prefix、functional LoRA替换及数据处理器；只新增一次性诊断入口，结束后退役。
物理source权重不更新。用于求导的同值weight叶只是计算变量；拟合及JVP只接demo16–19，不对诊断query反传。
无Writer/Meta训练、新预训练模型、数据复制、optimizer、checkpoint选择或RL。

2026-09-13已现场查询strg01：data1独立用量1016.7/1024GiB；analysis约37GiB、workspace临时约13GiB，共享空间充足。
本项新增峰值预算2GiB，包含约220MiB冻结源码树、192套约1GiB的完整rank16因子及小型预测／日志；source大权重与数据复用。
先用最小真实smoke核对梯度/JVP/LoRA作用与内存；正式诊断从clean pushed commit的detached frozen树运行。
launch前按gpu-preflight同时核对两节点，选一个节点至多6张实际提高吞吐的GPU；独立worker不使用NCCL，绑定GPU-local NUMA。
保存精确命令、commit、GPU分配、输出及退出状态；结束后核验原始行重算并关闭一次性入口，完整证据保留。

本项使用真实动作求一次任务条件更新，是明确的**训练侧oracle**。不得将它包装成action-hidden部署或以“只算一次”
绕开当前禁止task-local优化的边界。DAML的原论文Algorithm2也明确在meta-test使用梯度更新；它是相关机制，
并非当前部署合同的例外（[Yu等，RSS 2018](https://arxiv.org/html/1802.01557)）。
若后续考虑这种部署更新，必须先处理Owner科学边界；合法前向摊销生成则仍需证明自身获取与传递，不能借用oracle分数。
