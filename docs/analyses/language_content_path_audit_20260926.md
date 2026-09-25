# 语言内容路径：函数包含、学习信用与历史约束

2026-09-26。主讨论的理论/源码审计；没有新模型训练、forward或闭环结果。
本轮只授权末节的CPU构造验证，不授权新GPU计算。实际状态看progress。

## 1. 具体研究对象

解释同一Source/任务池/监督预算下，视频生成为什么没有稳定超过语言生成。
四臂共同节点C−B为−2、+12、−19、−13、−8、−19（各400，210至1260）。
因此“视频一加入就必然更差”已被该历史反驳；不把不同最佳节点的差值当单一架构效应。
Source到A/B/C/D的selected丢失分别9/10/24/18；能力交换不是视频Writer特有现象。
本项关注视频路径增加的表达/学习约束，不处理VLA普遍存在的闭环或FM问题。

## 2. 当前真实计算

源码owner：`writer/video_program.py`、`writer/temporal.py`、`writer/model.py`。
令q(L)为text-only原生读取和共享language_projection的结果；E(L,V)为逐帧多模态task-span与patch读取，
P(L,V)为完整H及相邻视觉证据经Procedure计算的结果。E本身含语言，不能说当前Core没有语义内容。

语言臂：

\[
 u_B=q(L),\quad C_B=T(u_B),\quad
 \Theta_B=H\bigl(F(N(\operatorname{Read}(C_B)),r)\bigr).
\]

视频臂：

\[
 u_C=A(q(L),E(L,V)),\quad C_C=T(u_C),
\]
\[
 c=\operatorname{Read}(C_C),\quad
 p=\operatorname{Read}_P(r+N(c),P-\bar P),\quad
 (\gamma,\beta)=W_mN(p),
\]
\[
 \Theta_C=H\bigl(F((1+\gamma)\odot N(c)+\beta,r)\bigr).
\]

T是共用的语言轴blocks，F是post_fusion，H是完整FactorHeads加identity template，r为slot routing。
A的query来自q，Value来自E；没有把q直接加到A的输出。frame gate混合均匀和学习attention，
不是“把video关掉就返回q”的残差开关。将其gate设为0仍读取视频均值。
这属于计算路径差别，不是当前模型不能表示语言函数的完备性反证。

## 3. 可验证的最小候选与两个构造性包含

仅在A之后、T之前新增一个共享标量a及直接内容项：

\[
 u_+=A(q,E)+a q,\qquad a_{init}=0.
\]

其余计算、输入和监督保持；a是所有task共享的可学习标量，不按task-ID切换。
这只是待检验的残差参数化，不称已确定新方法。

1. **包含原C。** a=0并复制原C所有参数，完整输出与C相同；新增标量不消费随机初始化。
2. **包含现有B。** a=1、A的最终output矩阵为0、W_m=0，且共用text reader、T、Core reader、
   routing、core_norm、post_fusion和heads取B参数，即有u_+=q、融合=N(c)，完整输出与B相同。
   E/P仍是合法视频计算，但其输出在该构造中不影响结果；不使用fake frame/native forward。
3. **不包含任意直接rank16的声明。** 八个共享head仍有限制。例如q_B各层B_l均位于同一个
   2048×216输出矩阵的列空间，其跨层合并列空间维数至多216；自由各层rank16没有同一限制。
   模板A、BA等价重参数化也不能无条件解除这个B列空间约束。B与C共同具有它，故不能用它单独解释C−B。
   该事实已在旧设计中登记，不是新发现的根因；当前private Writer并未输给freeAB的反例仍有效。

于是对相同输入域和完整策略函数族，有F_B∪F_C⊆F_+。若S为同一分布上的期望闭环成功率：

\[
 \sup_{f\in F_+}S(f)\ge\max(\sup_{f\in F_B}S(f),\sup_{f\in F_C}S(f)).
\]

这仅证明理想可达性，**没有证明有限训练改善，更没有证明视频提供额外能力**。
a=0只保持初始化时原C函数；开始更新a后，不能宣称原参数梯度与未来训练轨迹保持相同。
不由包含关系推出预训练B/公共LoRA后再接视频的课程；后续若训练，仍fresh联合学习。

## 4. 学习作用：可推导的差别和不能跳过的条件

令g_u=∂L/∂u_+。同一个实际参数点有

\[
 \partial L/\partial a=\langle g_u,q\rangle,\qquad
 \partial L/\partial q=(D_qA)^\top g_u+a g_u+\text{其它依赖q的路径}.
\]

新增项提供不经FrameRead注意力权重的内容/信用路径；不自动给出有益梯度。
对E固定的一头attention，设权重s_t=softmax(Q(q)^T k_t/√d)，v_t为Value、λ为均匀/学习混合gate：

\[
 D_qA|_E={\lambda\over\sqrt d}W_o
 \left[\sum_t s_t(v_t-\bar v_s)k_t^\top\right]D_qQ.
\]

该部分依赖Value/key的加权协变项；它可能小，也可能很有用。真实E还依赖语言和patch query，
完整导数须加D_EA·D_qE及共享参数项，不能由λ初值.05宣布总语言梯度被缩小20倍。
LoRA B-head零初始化也会暂时阻断上游信用，正常启用顺序不能解释数百步后的差距。

**工作假说H_path：** 在当前任务/预算中，任务内容必须经过视频条件化重编码这一参数化，
增加了语言到有效LoRA的学习代价；明确的内容路径可减轻该代价，同时保留可利用的video分支。
**竞争解释H_information：** 原video路径并非主要限制，监督未充分识别可迁移视频内容或其读取没有形成有用知识；
添加q至多恢复语言方案，不产生正确视频增量。两者允许并存。
另一个必须保留的风险是新增内容路径更容易拟合训练task，使优化更多依赖语言，进一步减少视频的有效学习。
因此C改善或a非零均不足以支持“视频学习改善”；若只到B水平、同task正确视频没有额外行为收益，
该结果同样符合语言捷径解释，不能靠模块存在或梯度非零排除它。
数学式只给计算机制和可达性，H_path所需的实际有益对齐尚无测量，不能把候选直接升级为修复。

## 5. 历史约束，不以“相容”冒充已解释

| 已有事实 | 对本候选的约束 |
| --- | --- |
| 旧v5.2 132/138/74、v6 121/122/111（correct/other/wrong） | 无直接q内容路径也能产生视频依赖；否定“该路径是视频有效的必要条件”。旧controls不调本候选，不能把其效应移植为新模型增量 |
| 当前C在共同420比B多12，后四节点均较低 | 不能预测全程统一劣势；后继须看共同节点及成功集合，不能只找一个candidate峰值 |
| 旧Compiler额外语言query删除曾改善，后来保持失败（E22–23） | 额外语言路由并不自动有益；这里增加的是T之前的Value内容，不是Compiler query仿射，差别需在实现中验证，收益仍待测 |
| 旧共享prior＋视频残差与多种新结构有能力损失 | 不宣称残差结构保证保持；本候选不继承公共LoRA、不相加两套LoRA、不做分阶段课程 |
| §128正确P关闭后仍有性能缺口，末帧修复不优于同预算原训练 | 不能只归罪Procedure；当前候选不调整P、时序、末帧或辅助loss |
| §129 private完整Writer16/32、freeAB14/32；缩head未修复 | 不把输出容量/步幅当已经识别的原因，不重跑旧局部可达性试验 |
| §143关系补齐无联合修复；§144读取之后W变化传递损害、N部分补偿 | W含多个模块，未单独定位Core；这只能支持继续检验具体结构，不是本残差的因果证据 |
| §145–150单slot、lookahead、回报修正未形成修复 | 不给新候选附加这些目标或优化器；保持普通已登记FM以解释结构主效应 |

本理论目前能排除若干过强说法，但尚不能解释全部具体分数及训练差异。
下一步要以针对性结果决定H_path是否值得保留，而非事后把任何结果解释成“共同适应”。

## 6. 行为分支和后续训练设计边界

- 新候选对原C、B及另一正确视频都有绝对收益，且没有更大的已有能力损失：才支持把此结构作为方法候选；
  仍需另一个有信息量节点和正式泛化/视频证据，不能只凭一个峰值。
- 仅恢复B的能力，正确视频没有额外收益：只登记结构代价可能减少，不称视频修复；不靠降低wrong满足要求。
- 候选没有比原C改善：不继续扫描a初值/尺度/层位，当前有界证据不支持该参数化；也不证明所有架构改动无效。
- 训练任务改善而诊断未见任务不改善：保留获取/迁移区别，不自动增加训练或把训练收益外推。

正式训练合同须把这些分支落实为固定节点、matched source/rank/task/query/label/optimizer时钟、
有限闭环与按需补样本。现有B/C原件可作为历史，但新的主要配对要处理正常self变化及实现/物理拓扑差异。
不在本CPU阶段确定新训练步数、预算或阈值；也不允许执行方自行从原1260合同恢复训练。

## 7. 本批CPU实施合同：language_content_path_cpu_20260926

实际执行者Sol：01a0cd90-ebb7-77a1-a20b-a858825d2f66；主讨论：01a0cd94-65da-7b22-8ca9-7ba35f454632。
目标仅为验证第3节构造能够在真实owner代码中成立，并给后续科学设计提供精确改动面。

- 从最新main隔离开发；主讨论拥有本文及三份状态文件，Sol只改必要源码/CPU验证，暂不集成或启动GPU。
- canonical原行为默认不变；候选只在FrameRead之后、language blocks之前加入a*q，共享标量零初始化。
  不改视频读取、P、heads、source、数据、loss、优化器或部署接口；不复制第二个Writer。
- 用现有真实Core/compiler/FactorHead模块和非零有效head参数验证：a=0复现C；第3节指定参数构造复现B；
  视频扰动仍能改变候选输出；非退化标量目标下a有正确梯度。以输出/梯度作用验证，不只检查参数存在。
  使用明确标记的合成中间表示，覆盖padding及不同token/frame长度；不是原生policy forward或行为证明。
- 不加载完整π0.5/source、不读视频/HDF5/action/state/outcome、不做Writer真实数据forward、bank或env.step；
  GPU进程、GPU-hours、正式query、闭环均必须为0。无需为CPU检查查询GPU状态或运行GPU preflight。
- FP32和正常容差即可；不扩dtype、追bitwise、全权重hash或建立大测试框架。最多一个小型针对性测试文件及必要既有回归。
- 若精确构造需第二项架构变化，或owner代码存在使第3节不成立的事实，保留反例并回报，不自行扩候选。
- 仅轻量CPU产物（≤16MiB）；不复制模型/data，不建立新大run root。开发代码工作树按既有Git/存储规则管理。
- 完成后保留独立分支/commit及小型结果，主动Queue主讨论：实际改动、数学构造读数、命令/exit、边界及异常；
  停止新增工作，不自动训练、profile、GPU smoke、集成main或接续旧flow实验。

CPU验证完成只证明实现可表达预定对照。是否值得训练由主讨论结合上述机制和历史继续裁决，不能据CPU通过宣称原因已找到。
