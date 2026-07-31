# EMBER Target-Spectral Writer

状态：2026-07-31 canonical fresh experiment。

## 1. 决策

当前最强合法结果仍来自 v5.2/v6，而新增的跨架构 LoRA 数值复核定位了一个
比继续重做上游更直接的瓶颈：

```text
v6 macro400 effective BA stable rank       1.000274
v6 q / v 的 B-column cosine                .99785量级
v6 18层 q / v effective-delta cosine       .969 / .984
direct Source-SFT跨层方向                  近零相关
```

该塌缩跨 v6、Core-Program、Prior、checkpoint、task 和 video 稳定存在。
视频产生的变化并非纯 scale；v6 macro400 同 task 多视频差异约 90.6% 位于
task-mean 的正交方向，但全部视频方差只占 LoRA 平均能量约 0.3%。因此上游
确实形成了视频方向创新，当前 decoder 却把它压在巨大的 task/common 更新中。

本版只做一个根本性模型修改：

> 保留 v6 已验证的 Core/Procedure 上游和融合职责，把 public-LoRA compiler
> 改为 target-first、rank-last，并固定 A/B spectral gauge。

首轮不同时改 optimizer。full-24 漂移可能同时来自任务梯度冲突和当前近全局
rank-1写入通道；先修复已有直接证据的 decoder，随后以逐任务 Gradient Gram
决定是否引入 conflict-aware update，避免两个变量同时改变。

## 2. 信息与训练合同

- Writer 输入仍为正确 task language 与恰好一条 action-hidden teacher video。
- 一条 video 只生成一套 LoRA；训练和推理均不做多视频平均。
- action queries 与 teacher video 只要求同 task，继续跨 episode 独立采样。
- teacher video 是 support；独立 action batch 是 query。该分离要求 Writer
  提取跨初态可迁移的方法，不能复制单条示范的绝对路径、速度或 phase。
- Procedure 不是单纯 source policy hypothesis 或环境结果：

```text
native Action-Expert hidden
    提供 source policy 的高层动作语义词汇

teacher-video task-grounded transition
    提供示范实际发生的对象、接触和目标关系变化

causal Procedure
    表达 teacher 展示的有序动作—环境变化过程
```

- 首轮训练仍为同点 full-24、每 task 一 video/LoRA、B20 独立 action queries、
  一次 AdamW update；不使用 checkpoint 融合、两阶段训练、contrast/order
  loss 或 reward。

## 3. 保留的 v6 上游

```text
task language + raw teacher frames
→ Q_text
→ multimodal task-token evidence M_f
→ task-query patch evidence G_f
→ X_f = M_f + G_f

X frame set
→ v6 permutation-invariant Semantic Core

native 50-suffix mean Action A_f
+ uncapped task-grounded D_f = G_f - G_(f-1)
→ two-layer causal Procedure
```

Core表达任务、对象角色、目标关系与跨帧不变量；Procedure表达teacher视频中
实际展示的有序动作过程。首轮不改Action probe、patch grounding、transition
或causal encoder。

## 4. 38个真实semantic targets

旧 compiler 把 `18 layers × rank16 + action_in/out × rank16` 组织成320个
semantic queries。rank index本质是代数分解坐标，不是教学语义；同时
layer/rank identity只进入attention Q/K，没有改变factor value/output basis。

新 compiler先只形成38个真实policy targets：

```text
18 q_proj + 18 v_proj + action_in + action_out = 38
```

每个target用独立routing读取Core；随后沿用v6 Core-primary Procedure AdaLN：

\[
C_m=\operatorname{CoreRead}(r_m,\mathrm{Core})
\]

\[
P_m=\operatorname{ProcedureRead}(r_m+\operatorname{Norm}(C_m),
                                 \mathrm{Procedure})
\]

\[
Z_m=(1+\gamma(P_m))\operatorname{Norm}(C_m)+\beta(P_m)
\]

`gamma/beta`保持zero-init，之后只在38个target层面执行一次content slot
coordination。rank尚未出现。

这里有意保留v6已经验证过的融合边界：Procedure query为
`routing + Norm(Core target)`，并允许打开后的`beta`直接写入teacher
Procedure创新。它不是“Procedure必须依赖非零Core才可贡献”的严格双必要
结构；Core-Program/Recenter已经证明该限制会损害absolute。另一方面，
Procedure value按时间中心化，所以恒定Procedure本身不贡献innovation，
此时仍保留Core semantic target。

## 5. 参数地址进入value

每个target拥有独立、bias-free、正交初始化的坐标变换：

\[
\hat Z_m=Z_m T_m,\qquad
T\in\mathbb{R}^{38\times256\times256}
\]

这使地址不仅决定“从memory读什么”，也决定“以哪个policy参数坐标写出”。
变换无bias，因此zero content不能由地址凭空生成adapter。

语义融合完成后才展开rank：

\[
H_{m,r}=\hat Z_m R_r,\qquad
R\in\mathbb{R}^{16\times256\times256}
\]

`T_m`和`R_r`均用不同signed-permutation orthogonal matrix初始化并可训练。
输出为：

```text
H: [batch, 38 targets, 16 ranks, 256]
```

rank展开后不再做跨rank self-attention。

## 6. 固定 A/B spectral gauge

每个target的16个hidden分别生成A方向残差、U方向残差和spectral scale：

```text
A_raw = A0 + ΔA(H)       [16, in]
U_raw = U0 + ΔU(H)       [out, 16]
s     = ScaleHead(H)     [16]
```

其中：

- `A0`由原PEFT template做FP32 reduced-QR并保留原平均row norm；
- `U0`是deterministic column-orthogonal carrier；
- A/U direction heads与4个type-specific scale heads的final Linear均
  zero-init。

输出：

\[
A=a_0\operatorname{RowOrth}(A_{\rm raw})
\]

\[
B=\operatorname{ColOrth}(U_{\rm raw})\operatorname{diag}(s)
\]

\[
\Delta W=BA
\]

step0时`s=0`，所以所有有效delta严格为零。模型仍可合法选择effective rank1，
但必须通过仅一个`s_r`非零来表达；不能再用16条相同A行/B列伪装rank16。
不加入rank-diversity或奇异值均匀loss。

## 7. 参数与首轮训练

真实枚举：

```text
Writer total          14,495,744
compiler               4,992,512
factor direction heads 2,179,072
spectral scale heads     263,168
```

相对原10–11M软预算有所提高，但新增容量全部对应已实测缺失的target/rank
parameter coordinates；公共宽度仍为256，rank仍为16。

首轮保持：

```text
GPU4–7
4 DDP ranks
每macro全局24 tasks
每task恰好1 video → 1 LoRA
B20 independent action queries
fast cosine decay400
fresh identity
macro0→200
every25 checkpoint
```

## 8. 判定

macro25先做无rollout结构门，固定8 validation tasks × 2 videos：

- q/v B-column cosine；
- representative cross-layer effective-BA cosine；
- A/U orthogonality；
- same-task video centered variance与正交方向占比；
- layer-energy CV，排除单层爆炸伪装specialization。

旧架构在macro50已经满足：

```text
B-column cosine    >= .997
cross-layer q      ~= .969
cross-layer v      ~= .984
```

若新架构仍同时出现：

```text
B-column cosine >= .995
cross-layer q   >= .95
cross-layer v   >= .97
```

则parameter address仍未真正生效，应停止而不是等待一小时碰运气。

结构门通过后评测macro50/100/150/200 paired correct400，只选择single
checkpoint。若达到150，或稳定接近并显著高于同期v5.2/v6，才对winner补
same/wrong/shuffled/reversed full400；否则只做内部数值与逐任务梯度分析。

## 9. 漂移后续门

本轮不预设full-24平均必然是根因。decoder修复后用单视频/B20 fixed macro记录
24-task Gradient Gram、cancellation ratio、原Adam候选预计伤害的任务数以及
Meta/Core/Procedure/compiler/factor分块冲突。

只有这些数据确认大量负冲突时，下一实验才采用单阶段Projected Full-24
update；不会同时引入多video、same-episode、两阶段或checkpoint aggregation。
