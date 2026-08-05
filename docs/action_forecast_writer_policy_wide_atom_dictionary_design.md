# Policy-Wide Atom Dictionary Writer 设计

状态：2026-08-05 fresh architecture authority。本文在SFT-Anchored Tangent-Basis
`143→142`且gained/lost=`20/21`的负消融后建立。新Writer必须从functional identity
fresh训练，不加载v6或任何历史Writer checkpoint；旧方法只由Git、design与formal
artifact保留。

## 1. 结论先行

下一条单变量架构不再让每个policy target/rank经共享factor MLP临时生成方向，也不把
v6已经学出的局部readout当固定答案。它显式学习一组跨全部38个policy targets对齐的
rank-1 policy atoms，再由语言和一条action-hidden视频生成两个低秩mixing matrices，
一次性把同一组atoms编译成公开rank-16 LoRA：

```text
task language + exactly one action-hidden teacher video
  -> v6 semantic Core + causal Procedure
  -> 16 condition-dependent policy-coordinate states
  -> M_A(x), M_B(x) in R^(16 x K)

K policy-wide atoms:
  D_A[target] in R^(K x input_width)
  D_B[target] in R^(output_width x K)

public A[target,x] = template_A[target] + M_A(x) D_A[target]
public B[target,x] = D_B[target] M_B(x)^T
```

首版固定`K=64`，即公开rank的4倍过完备容量。atom索引在全部q/v/action-in/action-out
targets之间共享；每个target保留自己的真实input/output向量。atoms不是可单独部署的
adapter bank，不读取task ID，也不按suite/task保存权重。部署时仍只生成并安装一套
完整rank-16 LoRA。

## 2. 为什么这是当前最早未检验接口

已有证据共同排除了四条更浅解释：

1. v6 macro400 correct143但effective BA近rank-1、视频写出弱；高absolute不证明
   generator健康。
2. Target-Spectral把stable rank抬到3.32却只得34，说明强制高rank/正交会破坏有用的
   coherent policy方向。
3. Direction Store与Target-Owned分别解除task store共享和policy-target head共享，仍只
   得129与99；参数分开不等于生成closed-loop有效方向。
4. Tangent-Basis冻结v6的8个factor-output readout后aggregate保持142，但41个state继续
   换手；局部basis旋转不是task drift的充分解释，而且warm-start不能修复v6已有流形。

direct rank-128 Source-SFT提供了正证据：每个target内部也主要低秩，但不同policy
layer/target具有稳定、非uniform的方向和能量组织。old/corrected SFT的q/v layer profile
高度复现，而Direction Store/SFB的不同layer常被共享factor decoder写成近同向。因此
当前缺失的不是更多独立heads或更漂亮谱，而是一个以**完整policy方向**为存储单位、
同时允许条件连续组合的坐标系。

## 3. 与历史方法的结构差异

- v6/SFB：存储单位是8个factor-family局部decoder；不同target只改变decoder输入。
- Direction Store：按语义route复制局部factor参数，仍没有policy-wide atom身份。
- Target-Owned：76个target独立输出，修复跨层同向但丢失跨target协调。
- Tangent-Basis：固定的是v6局部factor readout，不是从fresh objective学习的完整policy
  atom；结果只作消融。
- 本方法：每个atom从q0到q17、v0到v17及action projections拥有一套共同索引，条件
  mixing对全部targets一致。target-specific方向和能量存于atom内容，task/video差异存于
  mixing，不再在每个target内各自发明互不协调的条件方向。

## 4. 条件composer

保留已有、已验证的信息前端：

- text-only task queries；
- multimodal task-token与task-queried patch evidence；
- permutation-invariant mean-backed Semantic Core；
- fixed Action-Expert probe、task-grounded adjacent transition与causal Procedure。

删除320-slot compiler和8个factor heads。新composer只维护16个learned
policy-coordinate queries。每个query分别读取Core和Procedure：

```text
C_r = CoreCrossAttention(Q_r, Core)
P_r = ProcedureCrossAttention(Q_r + C_r, Procedure)
Z_r = GELU(W_z [C_r ; P_r])

M_A[r,k] = <W_A Z_r, key_A[k]> / sqrt(d)
M_B[r,k] = <W_B Z_r, key_B[k]> / sqrt(d)
```

Core与Procedure使用独立attention normalizer；Procedure已经由causal RoPE编码真实
采样帧顺序。没有softmax over atoms、top-k、task route、scalar gate、manual branch
scale、rank/orthogonality loss或额外监督目标。signed dot-product mixing允许模型自然
选择建设性同向、低rank或多atom组合。

## 5. Fresh identity与梯度阶段

`D_A`与`D_B`全部exact-zero，公开A仍是sealed random template，公开B为物理零。因此
step0逐tensor严格等于functional identity，不需要历史Writer初始化。

预期梯度阶段是模型数学本身而非训练trick：

1. 第一次functional backward只有`D_B`获得非零梯度；同一condition mixing把完整policy
   target梯度写入对齐atom索引。
2. `D_B`打开后，B-side mixing、composer、Core/Procedure可获得梯度，同时`D_A`开始打开。
3. `D_A`非零后，A-side mixing也可学习；后续所有前端与dictionary参数共同端到端训练。

profile必须真实观察该阶段，不能用手工非零B、warm-start或额外loss跳过。

## 6. 为什么可同时用于AS与RL

AS functional objective和rollout reward只是在同一最终LoRA上的两种credit来源；架构没有
teacher-action专属输出、LIBERO规则或监督辅助head。首阶段从identity用24 train tasks的
合法action functional loss同时学习policy atoms与condition composer。只有fresh AS本身
证明absolute、视频传递和dictionary使用健康后，才允许在同一checkpoint轴关闭action入口，
冻结或慢更新atoms并用reward继续校准mixing。RL不能用来挽救一个未通过AS机制门的
generator。

## 7. 首轮训练合同

保持现有A40科学口径，不把架构与recipe再次混杂：

- 24 train tasks完整full24 raw mean；
- 每task exactly one video、logical B20 independent action queries；
- six ranks、每rank 4 tasks、policy physical microbatch2；
- AdamW、fast-decay400、每25保存；
- source policy和normalization冻结；
- validation/test action gradient为0；
- fresh0→200后评测50/100/150/200 strict paired correct400；只有absolute、breadth、
  success-set retention、趋势和内部路径共同支持才续200→400。

首轮不加入reward、policy anchor、multi-video、contrastive/order loss、dictionary load
balance、atom orthogonality、谱约束或checkpoint融合。

## 8. 实现与profile门

canonical `CompleteLoRAWriter`原位替换compiler/factor输出，历史v6可执行decoder由Git
保存；新增一个凝聚的policy-dictionary owner，不保留双路runtime。聚焦CPU合同至少覆盖：

1. 38 target shapes、K64与rank16 mixing精确；
2. fresh output逐tensor等于template；
3. 所有target共享同一mixing但保留不同atom参数；
4. condition改变后mixing与生成LoRA改变；
5. step1/2/3梯度阶段与全路径finite；
6. source trainable=0和信息墙不变；
7. checkpoint schema拒绝v6/历史Writer权重。

真实六卡profile必须覆盖最长105-frame、logical B20、完整full24三步、0 OOM/nonfinite，
并证明dictionary、composer、Core、Procedure按预期打开。profile checkpoint永久弃用；
正式run必须来自clean/pushed代码的独立fresh root。

## 9. 判定

closed-loop仍是唯一性能裁决。内部重点报告：atom mixing的task/video variance、有效atom
participation、不同target的energy profile、effective BA谱、same-task多video差异以及
fixed-query action传递，但不为这些指标设置人工“越高越好”门。

- 若mixing随视频变化且到BA/action有效，但correct低：dictionary方向仍off-manifold，
  下一步改policy-atom学习目标或更新时钟，不扩大K掩盖。
- 若dictionary有功能但mixing跨task持续互换：问题在condition composer credit，才考虑
  atom固定后的AS/RL分阶段或语义拥有权。
- 若fresh方法同时保留v6 absolute并接近v5.2视频干预margin，再进入reward校准。
- 若K64本身没有被使用或全路径不能打开，直接负裁决，不用K128、scale或额外loss救活。

## 10. 当前实现状态（2026-08-05）

canonical实现、新config与独立launch/checkpoint family已完成；旧320-slot compiler/factor
decoder已从活动Writer源码删除。CPU聚焦合同通过41项，确认13,033,728参数、全部38 targets
形状、fresh identity、conditioned LoRA以及真实BA functional loss的梯度阶段。

clean`60e45f8`已完成BCI六卡longest105/logical-B20三步profile：step seconds=
`32.860/30.418/30.404`，峰值allocated/reserved=`35,024,829,440/44,883,247,104`
bytes，0 OOM/clip；1,440 queries、72 one-shot videos，step1仅policy atom，step2起五个
声明block全可达，source trainable=0。独立fresh0→1后首次resume因新family漏入optimizer
restore合法集合而fail-fast，0新增metric/checkpoint；最小修复后原六卡step1→3重放通过，
六rank state与optimizer/scheduler/RNG/data cursor闭合。formal config现已seal，下一边界是
clean/push与live preflight后的独立fresh0→200；任何profile/smoke权重均不得进入。
