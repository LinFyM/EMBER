# EMBER Loom：Task-Grounded Teacher–Policy Gap Writer

状态（2026-07-30）：本文是历史Loom规格，只作provenance。Loom已实现并完成
首段；macro50/100/150/200 correct400为`79/106/105/112`，内部
correspondence/confidence/Teacher–Policy gap缺少信息墙内可靠锚点，因此停止。
活动源码不再保留Loom路径。

当前唯一canonical设计为
[`action_forecast_writer_recenter_design.md`](action_forecast_writer_recenter_design.md)。
不得从Loom checkpoint resume，也不得恢复Loom/Recenter双路径。

名称 **Loom** 表示：

```text
Semantic Core                 稳定的经线：任务、对象、目标、约束
Teacher / Policy Procedures   有序的纬线：teacher过程与source模仿假设
Gap Compiler                  只把source真正缺失的知识织成public LoRA
```

完整描述为：

> **EMBER Loom: Task-Grounded Teacher–Policy Gap Writer**

## 1. 最终目标与第一性原理

Writer 的理想输出不是“teacher视频的编码”，而是当前 source policy 相对
teacher 展示过程仍缺少的能力：

\[
\Delta\theta
=
\operatorname{Adapt}
\left(
\text{task},
\text{teacher procedure},
\text{source-policy procedure}
\right).
\]

概念上：

```text
teacher展示了什么
- source policy已经会什么
= LoRA仍需要补充什么
```

所以最终架构必须显式得到：

1. **Semantic Core `C`**
   - 任务是什么、对象是谁、目标和约束是什么；
2. **Teacher Procedure `P_T`**
   - 视频实际展示了什么高层运动、环境效果和目标进展；
3. **Policy-Imitation Procedure `P_A`**
   - Action Expert 站在 teacher 视频各状态上，会怎样理解和模仿该过程；
4. **Teacher confidence `q`**
   - 当前视频证据是否任务相关、匹配一致且时序连贯；
5. **Teacher–Policy gap `d`**
   - 可信 teacher procedure 与 source imitation hypothesis 之间的高层差距。

最终 LoRA 的内容来自 `d`，强度由 `q` 与 gap magnitude共同决定。Core只帮助
解释差距，Action不能独立生成adapter，teacher变化能量也不能直接等同于教学
价值。

## 2. 信息墙与不变输出合同

Writer 输入严格为：

```text
exact task language
+ exactly one ordered raw action-hidden teacher video
```

Writer 不得读取：

- teacher action；
- teacher proprio/state；
- reward、success、terminal；
- task/suite ID、filename、episode identity；
- object pose、hidden normalization或policy outcome；
- 可反推出上述信息的元数据。

teacher actions只进入frozen source policy下的AS functional loss。video与action
query在同一task内独立采样，不要求同episode或逐帧配对。

输出保持一套完整sealed public rank-16 task LoRA：

```text
38 targets
76 A/B tensors
320 routing slots
rank 16
alpha 16
dropout 0
```

frame stride、source base、normalization、24/8/8 split和official evaluation
接口保持项目authority规定不变。

## 3. 总体数据流

```text
task language T
        │
        ├─ text-only Gemma + Text Meta-LoRA
        │      → stable task queries Q_text [L,256]
        │
teacher frame I_f
        │
        ├─ multimodal Gemma + VL Meta-LoRA
        │      ├─ task-token evidence M_f [L,256]
        │      └─ patch evidence P_f [N,256]
        │
        ├─ Q_text reads P_f
        │      → task-grounded patch evidence G_f [L,256]
        │
        ├─ X_f = M_f + G_f
        │      ├─ frame-set semantic fusion
        │      │      → Semantic Core C [L,256]
        │      └─ adjacent semantic relations R^L_f
        │
        ├─ P_f ↔ P_(f+1) bidirectional correspondence
        │      → visual motion/effect relations R^V_f
        │
        └─ same prefix KV
               + 8 sparse Action suffix probes
               + Action Meta-LoRA
               → Action hypotheses A_f [8,256]

[R^L_f, R^V_f]
        → task relevance + bounded matching confidence
        → 8 Teacher Event tokens E_f
        → Teacher confidence q_f

E_0...E_(F-2)
        → shared axial temporal operator
        → Teacher Procedure P_T

A_0...A_(F-1)
        → same shared axial temporal operator
        → Policy-Imitation Procedure P_A

320 routing identities
        → read Core C
        → separately read P_T and P_A
        → learned high-level alignment
        → d = teacher - policy
        → r = teacher confidence × bounded gap magnitude
        → Core-assisted gap content
        → content-only slot coordination
        → eight factor heads
        → ΔLoRA = r × factor output
        → complete public rank-16 LoRA
```

## 4. Foundation特征与Meta-LoRA

### 4.1 三套Meta-LoRA

保持已在v5.2/v6中证明有价值的三套rank-4 Meta-LoRA：

| module | role |
|---|---|
| Text Meta-LoRA | 产生video-independent稳定task query axis |
| VL Meta-LoRA | 从每帧图像与语言提取task/patch evidence |
| Action Meta-LoRA | 让原生Action Expert适配8-token sparse suffix probe |

三者均为正常小随机`A`、物理零`B`初始化；frozen source主干不更新。

Action Meta-LoRA不能删除。原始Action Expert适配50-token suffix，Loom只输入
8个原生位置；Meta-LoRA必须学习怎样用稀疏probes询问source Action Expert。

### 4.2 Text-only task queries

exact task language单独经过frozen Gemma + Text Meta-LoRA，再用共享bias-free
`2048→256` projection：

\[
Q_{\text{text}}\in\mathbb{R}^{L\times256}.
\]

它只提供稳定task token identity、语义与语言顺序，不携带视频内容。

### 4.3 每帧multimodal evidence

每帧只做一次multimodal prefix forward，得到：

```text
H_task[f]  ∈ R^(L×2048)
H_patch[f] ∈ R^(N×2048), current N=256
prefix KV
```

共享bias-free `2048→256` projection：

\[
M_f=W_{\rm sem}H_{\rm task}[f]\in\mathbb{R}^{L\times256},
\]

\[
P_f=W_{\rm sem}H_{\rm patch}[f]\in\mathbb{R}^{N\times256}.
\]

task/image tokens本来处于同一Gemma hidden space，因此共享projection；各读取
模块可有独立RMSNorm，但不创造两个任意语义坐标系。

### 4.4 为什么保留task-queried patch readout

Gemma内图文已经交互，额外readout不增加原始信息；但“信息可能存在”不等于
Writer在task-token axis上有稳定、可访问的patch value path。

v5.1只读取multimodal task positions时上游语义不足；v5.2加入稳定
task-query→patch-value路径后同时改善absolute与空间/对象证据。v6的最高合法
single-checkpoint性能也建立在这一路径上。v10失败的内部证据定位到Action主导
与compiler增益，而不是Core/grounding失效。

因此该readout的必要职责是：

```text
用video-independent task identities选择patch content
→ 把真实patch values显式投影到稳定task-token axis
```

每帧：

\[
G_f
=W_o^G\operatorname{Attention}
\left(
W_q^G\operatorname{RMSNorm}(Q_{\rm text}),
W_k^G\operatorname{RMSNorm}(P_f),
V=P_f
\right).
\]

配置：

```text
width       256
heads       8 × 32
Wq/Wk/Wo   bias-free
Wv          none
```

完整逐帧task-grounded semantic evidence：

\[
X_f=M_f+G_f\in\mathbb{R}^{L\times256}.
\]

这不是第二套视觉encoder，也不是声称“重新创造图文信息”；它是一个显式、稳定、
有唯一value owner的task evidence readout。

### 4.5 八个Action probes

Writer-only suffix使用原生50-position horizon中的：

```text
[0, 7, 14, 21, 28, 35, 42, 49]
```

合同：

- 一次Action Expert forward；
- suffix sequence length为8；
- 保留原生position IDs与对应固定Gaussian rows；
- 不是8次forward；
- 不改变execution policy原生50-action chunk；
- 取final norm后、`action_out`前的hidden。

经共享bias-free `1024→256` projection：

\[
A_f\in\mathbb{R}^{8\times256}.
\]

`A_f`不是teacher action，也不是严格冻结的base成功率测量。它表示：

> 经8-probe接口适配后，source Action Expert面对teacher当前状态产生的模仿
> hypothesis。

## 5. Semantic Core

Core必须回答：

```text
任务是什么？
涉及哪些对象、角色、目标关系与约束？
语言声明了几个子目标及其顺序？
哪些任务相关场景条件跨执行过程保持稳定？
```

它表达高层语义身份，不依赖固定像素位置；对frame set排列严格不变，但保留
task-token ordinal。

### 5.1 Mean backbone

对每个有效task token：

\[
\mu_l=\operatorname{Mean}_f X_{f,l},
\qquad
\Delta X_{f,l}=X_{f,l}-\mu_l.
\]

mean是不可丢失的静态语义骨架。

### 5.2 Task-conditioned centered residual

使用稳定`Q_text,l`选择偏离mean的frame evidence：

\[
R_l^C
=W_o^C\operatorname{Attention}_f
\left(
W_q^C\operatorname{RMSNorm}(Q_{{\rm text},l}),
W_k^C\operatorname{RMSNorm}(X_{f,l}),
V=\Delta X_{f,l}
\right).
\]

\[
U_l=W_m^C\mu_l+R_l^C.
\]

性质：

- 对frame permutation严格不变；
- mean backbone不会因attention训练失败而消失；
- attention若在frame上均匀，centered value精确相消；
- task query只选择真实video values，不能凭空注入content。

### 5.3 Language-axis composition

`U∈R^(L×256)`进入两层bidirectional task-token Transformer：

```text
width       256
heads       8 × 32
FFN         256 → 1024 → 256
depth       2
norm        pre-RMSNorm
position    task-token ordinal RoPE, Q/K only
dropout     0
```

输出：

\[
C\in\mathbb{R}^{L\times256}.
\]

保留`L`个Core tokens，使容量随对象、目标关系和多子目标复杂度自然变化。Core
不读取frame order，也没有Core→factor-head直接路径。

## 6. Teacher Event原材料

Teacher Procedure必须来自视频实际展示的变化，而不是Action hypothesis。

### 6.1 Task-semantic interval relation

对当前arm实际输入顺序中的相邻帧：

\[
\bar X_{f,l}=(X_{f,l}+X_{f+1,l})/2,
\qquad
\Delta X_{f,l}=X_{f+1,l}-X_{f,l}.
\]

\[
R^L_{f,l}
=W_o^L
\left[
W_d^L\Delta X_{f,l}
+
\tanh(W_g^L\Delta X_{f,l})
\odot W_m^L\bar X_{f,l}
\right].
\]

所有linear为bias-free `256→256`，因此：

```text
Delta X = 0 → R^L = 0
```

绝对task content不能通过semantic relation形成静态Procedure旁路。

### 6.2 Bidirectional patch correspondence

对`P_f`与`P_(f+1)`做共享8-head cross-frame matching。二维patch coordinate
只进入Q/K：

\[
\pi^\rightarrow_{i,j}
=
\operatorname{softmax}_j
\frac{q(P_{f,i},p_i)^\top k(P_{f+1,j},p_j)}{\sqrt{32}}.
\]

\[
\widehat P_{f+1|i}
=\sum_j\pi^\rightarrow_{i,j}P_{f+1,j},
\qquad
\Delta p_i^\rightarrow
=\sum_j\pi^\rightarrow_{i,j}(p_j-p_i).
\]

反向使用同一参数得到
`\pi^←, \widehat P_(f|j), Delta p^←`。

定义forward mutual consistency：

\[
m_i^\rightarrow
=
\sum_j\pi^\rightarrow_{i,j}\pi^\leftarrow_{j,i}.
\]

并记录normalized matcher entropy。uniform或forward/backward不一致的matching
不能获得高confidence。

### 6.3 Same-grid、matched与displacement change

\[
d_i^{grid}=P_{f+1,i}-P_{f,i},
\]

\[
d_i^{match}=\widehat P_{f+1|i}-P_{f,i}.
\]

位移用32维zero-at-origin Fourier encoding：

\[
\psi(0)=0
\]

即sine与`cos(x)-1`，避免零位移产生常数content。

\[
\chi_i=
\left[
d_i^{grid};
d_i^{match};
\psi(\Delta p_i)
\right],
\qquad
\bar P_i=(P_{f,i}+\widehat P_{f+1|i})/2.
\]

\[
R^{V\rightarrow}_{f,i}
=W_o^V
\left[
W_d^V\chi_i
+
\tanh(W_g^V\chi_i)\odot W_m^V\bar P_i
\right].
\]

反向同理得到`R^(V←)`，参数共享。若grid/matched/displacement全为零，visual
relation严格为零。

该路径显式保留：

- 机械臂与末端执行器运动；
- 夹爪开合；
- contact与release；
- object co-motion；
- 搬运、遮挡、出现/消失；
- 目标关系形成。

它不加入optical flow、tracker、object detector、3D reconstruction或第二套
视觉encoder。

## 7. Relation content与teacher confidence

v10中shuffled visual-transition RMS约为correct的2.5倍，所以“大变化”不能
等同于“高教学价值”。Loom把relation direction与evidence confidence分离。

### 7.1 Bounded change presence

对relation token `R_i`：

\[
s_i
=
\frac{\operatorname{RMS}(R_i)}
{\operatorname{RMS}(R_i)+\tau_{\rm change}}
\in[0,1).
\]

content使用：

\[
\widehat R_i
=
\frac{R_i}{\operatorname{RMS}(R_i)+\epsilon}.
\]

因此超大shuffled跳变不会无限增大content或prior；变化幅度只负责确认“发生了
变化”，随后饱和。

### 7.2 Task relevance

对visual relation，用stable task query只计算相关性，不提供value：

\[
z_i
=
\operatorname{LSE}_l
\frac{
(W_q^TQ_{{\rm text},l})^\top
(W_k^T\widehat R_i)
}{\sqrt{32}}
-\log L,
\]

\[
t_i
=
\tanh
\left(
\operatorname{ReLU}(z_i/\tau_{\rm rel})
\right)
\in[0,1).
\]

减去`\log L`建立uniform/zero-similarity baseline；zero或负的task match得到
`t_i=0`，而不是普通sigmoid在无匹配时仍给`0.5` confidence。

semantic relation已与task-token position对齐，其task relevance由valid token
mask与relation content承担。

### 7.3 Matching confidence

visual relation confidence结合：

```text
bounded change presence s_i
× forward/backward mutual consistency m_i
× non-uniform matcher confidence h_i
× task relevance t_i
```

得到：

\[
c_i\in[0,1].
\]

其中non-uniform matcher confidence显式为：

\[
h_i
=
1-
\frac{\mathcal H(\pi_i)}{\log N}
\in[0,1].
\]

semantic relation使用bounded change presence与valid/task alignment得到`c_i`。
任何confidence factor都不能携带vector content。

### 7.4 为什么confidence不读取Action

teacher evidence是否可信必须由teacher视频、task和匹配一致性决定。若Action
参与`q`，Action Meta-LoRA又可自行决定是否获得adapter权限，重建v10捷径。

所以：

```text
q_teacher = f(task, teacher relations, temporal coherence)
```

严格不读取`A_f`或`P_A`。

## 8. 八个Teacher Event tokens

每个interval的relation集合：

```text
R_f = [
    R^L_f;
    R^(V→)_f;
    R^(V←)_f
]
```

### 8.1 三个evidence backbones

semantic、visual-forward、visual-backward各形成一个确定性backbone：

\[
q_f^{type}
=
\frac{\overline c_f^{type}}
{\overline c_f^{type}+\tau_{\rm type}},
\qquad
\overline c_f^{type}
=
\frac{1}{N_f^{type}}\sum_{i\in type}c_i,
\]

\[
B_f^{type}
=
q_f^{type}
\frac{\sum_{i\in type}c_i\widehat R_i}
{\sum_{i\in type}c_i+\epsilon}.
\]

这里第二个乘法是scalar confidence乘normalized relation aggregate。它保证：

- 不依赖learned selection也保留三类证据；
- confidence按有效relation的均值而非总数计算，不因patch/token较多而天然更高；
- 所有relations为零时backbone严格为零；
- 大energy只饱和，不会自动压过correct关系；
- uniform/inconsistent matcher被confidence降权。

### 8.2 五个learned relational events

另外五个learned queries读取全部relations：

\[
\alpha_{k,i}
=
\operatorname{softmax}_i
\left(
\frac{(W_qq_k)^\top(W_k\widehat R_i)}{\sqrt{32}}
+
\log(c_i+\epsilon)
\right).
\]

\[
q_{f,k}=\sum_i\alpha_{k,i}c_i,
\]

\[
H_{f,k}
=
q_{f,k}
W_o\sum_i\alpha_{k,i}W_v\widehat R_i.
\]

learned query、relation type与position只进入Q/K；value只来自normalized
relation content。若所有`c_i=0`，learned Event严格为零。

最终：

\[
E_f
=
[B_f^L,B_f^{V\rightarrow},B_f^{V\leftarrow},H_{f,1},\ldots,H_{f,5}]
\in\mathbb{R}^{8\times256},
\]

并保留对应：

\[
q_f^0\in[0,1]^8.
\]

没有learned null token，也不把八个Events提前压成一个。

## 9. Policy-Imitation Action stream

每帧八个Action tokens完整保留：

\[
A_f\in\mathbb{R}^{8\times256}.
\]

不进行：

- 8→1 mean或phase mixer；
- joint `8×L` competition；
- 局部Action×Effect multiplication；
- Action对Event value的直接写入；
- Action→LoRA独立旁路。

Action stream的职责不是提供teacher content，而是形成：

> source Action Expert沿teacher状态序列产生的高层模仿假设。

它将在完整Procedure层面与Teacher stream比较。局部`A_f,k`不被解释为造成
相邻视觉变化的teacher action。

## 10. Shared dual-stream axial Procedure

Teacher Events与Action hypotheses使用相同宽度和8个稳定slot identities，但在
比较前保持value memories分离。

### 10.1 输入

```text
Teacher stream:
E_0...E_(F-2)          [F-1, 8, 256]

Policy stream:
A_0...A_(F-1)          [F,   8, 256]
```

stream-specific input RMSNorm/projection和Q/K type embeddings保留；核心axial
block参数在两条stream间共享，使两类序列学习相同的高层时序算子，同时不混合
value content。

### 10.2 两层axial causal blocks

每层依次执行：

1. **local slot attention**
   - 在每个frame/interval内部沿8个slots双向计算；
2. **slot-wise temporal attention**
   - 对每个固定slot identity沿时间做causal attention；
3. **FFN**
   - `256→1024→256`，GELU。

统一配置：

```text
width       256
heads       8 × 32
depth       2
norm        pre-RMSNorm
dropout     0
linear      bias-free
```

sampled-frame ordinal、stream type和slot identity只进入Q/K，不作为value。

输出：

\[
P_T\in\mathbb{R}^{(F-1)\times8\times256},
\]

\[
P_A\in\mathbb{R}^{F\times8\times256}.
\]

两条memory不在Procedure encoder内相加、concat后self-attend或互相读取value。

### 10.3 复杂度

原混合草案对`16F-8` tokens做dense global attention：

\[
O((16F)^2).
\]

axial operator为：

\[
O(F\cdot16^2+16\cdot F^2).
\]

最长`F=105`时仍完整保留8个Action和8个Event slots，不需要单carrier瓶颈，
也避免相对v10约64倍的dense attention增长。真实B20/B16仍必须重新profile。

### 10.4 Temporal teacher confidence

Event初始confidence `q_f^0`随teacher stream保留为独立scalar。teacher temporal
output产生只可降低或重分配初始confidence的bounded coherence factor：

\[
q_{f,k}^T
=
q_{f,k}^0
\cdot
\sigma
\left(
w_q^\top\operatorname{RMSNorm}(P_{T,f,k})
\right).
\]

因为`q^0=0→q^T=0`，Temporal不能在无relation evidence时凭空制造可信度。
该head不读取Action stream。

## 11. 320-slot Teacher–Policy Gap Compiler

### 11.1 Routing identities

保持：

```text
18 expert layers × rank16     288
action_in rows                 16
action_out rows                16
total                         320
```

routing identity只进入Q/K，不进入value content。

### 11.2 Core read

\[
C_s
=
\operatorname{CoreRead}(Q=R_s,K=C,V=C).
\]

`C_s`回答该public LoRA slot当前面对什么任务对象、目标与约束。

### 11.3 Teacher read与confidence read

\[
T_s
=
\operatorname{TeacherRead}
\left(
Q=R_s+\operatorname{RMSNorm}(C_s),
K=P_T,
V=P_T
\right).
\]

使用同一teacher attention weights读取scalar confidence：

\[
q_s
=
\sum_{f,k}\alpha^T_{s,f,k}q^T_{f,k}
\in[0,1].
\]

`q_s`只来自task、teacher relations和teacher temporal coherence。

### 11.4 Policy-imitation read

teacher target帮助选择source hypothesis中应比较的部分：

\[
A_s
=
\operatorname{ActionRead}
\left(
Q=R_s+\operatorname{RMSNorm}(C_s)+\operatorname{RMSNorm}(T_s),
K=P_A,
V=P_A
\right).
\]

Action value只能进入这一步的comparison representation；没有Action→factor
旁路。

### 11.5 Learned high-level alignment

用两个bias-free full-rank projections映射到共同Procedure space：

\[
t_s
=
\operatorname{RMSNorm}(W_TT_s),
\]

\[
a_s
=
\operatorname{RMSNorm}(W_AA_s).
\]

二者比较的是Core-conditioned完整高层Procedure，不是局部7D action或严格同步
帧对。

Teacher–Policy gap：

\[
d_s=t_s-a_s.
\]

当二者归一化后，`RMS(d_s)`主要反映方向不一致，与cosine similarity单调相关。

### 11.6 Bounded adaptation strength

\[
g_s
=
\frac{\operatorname{RMS}(d_s)}
{\operatorname{RMS}(d_s)+\tau_{\rm gap}}
\in[0,1).
\]

\[
r_s=q_s\cdot g_s.
\]

含义：

```text
teacher可信，source imitation接近teacher
→ gap小，少适配

teacher可信，source imitation与teacher不同
→ gap大，强适配

teacher不可信或不相关
→ q小，无论Action怎样都接近identity
```

Action Meta-LoRA不能用幅度操纵`g`，因为`t/a`先归一化；也不能操纵`q`，因为
teacher confidence不读取Action。

### 11.7 Core-assisted gap content

\[
Z_s
=
\operatorname{RMSNorm}(d_s)
+
\tanh
\left(
W_g\operatorname{RMSNorm}(d_s)
\right)
\odot
\operatorname{RMSNorm}(W_cC_s).
\]

Core帮助解释“差的是哪个对象、目标和LoRA功能”，但只有最终`r_s`非零时才能
形成public update。

### 11.8 Content-only slot coordination

`Z∈R^(320×256)`进入一个slot block：

```text
Q/K = routing identity + normalized gap content
V   = gap content only
width 256
heads 8 × 32
FFN 256 → 1024 → 256
depth 1
```

routing identity不产生content。slot block可以协调一套完整adapter，但不能改变
每个slot单独保存的最终scale `r_s`。

## 12. Factor heads与public LoRA

保持v6已证明容量充足、硬件友好的八个factor heads：

```text
256 → 256 → target_width
GELU
all linear bias-free
final Linear zero-init
```

target widths：

```text
q_A / q_B / v_A / v_B             1024 / 2048 / 1024 / 256
action_in_A / action_in_B           32 / 1024
action_out_A / action_out_B       1024 / 32
```

对slot `s`，factor head先生成normalized gap direction，再在所有RMSNorm、FFN
和slot coordination之后显式乘回：

\[
\Delta {\rm factor}_s
=
r_s\cdot\operatorname{FactorHead}(Z_s).
\]

这建立连续合同：

```text
teacher evidence → 0
或 Teacher–Policy gap → 0
→ public LoRA delta连续→0
```

而不只是“输入精确为零时identity”。

输出写入既有deterministic functional-identity template：

- template `A`为确定性非零basis；
- template `B`为物理零；
- factor final zero-init；
- fresh step0完整public LoRA逐tensor、逐功能等于source identity。

## 13. 关键结构合同

实现必须满足：

1. **Core permutation invariance**
   - 同一frame set任意排列产生相同Core。
2. **Arm-local recomputation**
   - shuffled/reversed先重排raw frames，再重算Gemma states、relations、
     matching、Events和Procedures。
3. **Zero relation**
   - identical adjacent evidence产生零semantic/visual relation。
4. **No positional content**
   - positions、types、slot/routing identities只进入Q/K。
5. **Teacher/Policy value separation**
   - `P_T`与`P_A`在compiler comparison前不混合value。
6. **No Action-only LoRA**
   - teacher confidence为零时，任意Action输入均产生identity delta。
7. **No Core-only LoRA**
   - teacher confidence或gap为零时，Core不能生成delta。
8. **No false local causality**
   - 不宣称单个Action probe造成单个visual change。
9. **Continuous identity**
   - `q→0`或`d→0`时LoRA连续趋近identity。
10. **No raw-energy preference**
    - shuffled大跳变不能仅凭RMS获得更大teacher confidence。
11. **Full evidence preservation**
    - 8个Action与8个Event slots都参加local及temporal计算，不提前8→1。
12. **Action Meta interface**
    - 8-token suffix必须经过Action Meta-LoRA，保留native IDs和Gaussian rows。

## 14. 历史失败与成功能力审计

### 14.1 v4：保留原始思想，删除错误实现

v4有价值的核心思想：

```text
让source Action Expert站在teacher状态上想象“如果是我会怎么做”
```

Loom保留为`P_A`。

v4失败路径：

- 低层50×7 forecast；
- 未校准absolute robot clock；
- Plan/Revision假设；
- raw-image/Meta绕过visual-state；
- translation/phase latent直接成为controller。

Loom通过task-grounded teacher Event作为必要参照、Procedure-level gap和
Action-only禁路删除这些失败；不恢复旧forecast/Plan/Revision。

### 14.2 v5.2：继承最理想的视频语义门

v5.2五臂：

```text
132 / 138 / 74 / 82 / 83
```

有效优势：

- video-independent task query；
- task-queried patch values；
- permutation-invariant Core；
- causal Procedure；
- strong slot-specific LoRA transmission；
- wrong/shuffled/reversed接近base而非破坏base。

Loom全部保留这些功能，并用teacher confidence/gap进一步明确无效视频为何回到
identity。

### 14.3 v6：继承当前最强合法single-checkpoint能力

v6 task-complete fast-decay single-checkpoint best：

```text
correct / same / wrong / shuffled / reversed
143     / 135  / 125   / 128      / 129
```

v6最重要的有效优势：

1. stable task-grounded patch trajectory；
2. mean backbone + task-conditioned centered residual Core；
3. pretrained Action hypothesis没有被删除；
4. visual transition补充动态证据；
5. 两层高效causal temporal reasoning；
6. 320-slot strong compiler；
7. full-width factor heads；
8. exact identity initialization；
9. hardware-friendly width256、8×32。

Loom逐项继承：

- `Q_text/M/G/X`与v6语义前端；
-同样的Core集合骨架；
- 8-probe Action Meta path保留更多而非更少Action信息；
- semantic + raw visual transition扩展v6 transition；
- axial两层时序保留效率和完整证据；
- 320-slot与factor capacity不压缩；
- identity与硬件维度不变。

Loom只删除v6中后来被证据证明不理想的部分：

- Action作为独立Procedure content；
- raw transition能量缺少可信度解释；
- compiler增益不能区分source已会/未会。

### 14.4 v7：避免局部pair不可识别与Core被忽略

```text
joint 8×L entropy / uniform ≈ 99.96%
fixed-P, vary-C LoRA L2     ≈ 0.1–0.2%
```

Loom不做局部Action–Effect pair normalization；comparison发生在完整高层
Procedures之后。Core除query外显式进入gap content，但最终仍受`r_s`授权。

### 14.5 v8：避免Effect-dominant单Event

```text
EventRead entropy / uniform       ≈ 99.67%
vary Action, fixed Effect L2      ≈ 8–10%
fixed Action, vary Effect L2      ≈ 147–300%
```

Loom不做8→1 EventRead，不把Action严格乘进Effect；8个Action和8个Events在
完整时序中保留，最终通过Procedure gap比较。

### 14.6 v10：消除Action主导与尺度放大

v10正式best：

```text
correct / same / wrong / shuffled / reversed
103     / 94   / 75    / 67       / 43
```

内部：

```text
fixed Effect, vary Action LoRA    .6299 / .8659
fixed Action, vary Effect LoRA    .0808 / .1004
Effect attention entropy          ≈ 99.86% uniform
Procedure slot RMS                .0145
gated-Core RMS                    .1781
```

Loom的对应修复：

- Action不再进入teacher content或独立LoRA path；
- Teacher/Policy memories分离；
- gap而非Action本身决定适配内容；
- teacher confidence不读取Action；
- relation confidence不用raw energy；
- `r_s`在所有normalization之后乘回；
- source已会的公共能力在teacher-policy gap中自然抵消。

### 14.7 已验证优势的继承矩阵

Loom不是因为旧版本失败就整体另起炉灶。它把当前最强合法absolute架构v6和
当前最理想视频特异性架构v5.2中已经被结果支持的能力逐项指定了新owner：

| 已验证优势 | 证据来源 | Loom中的owner | 相比旧版新增的约束 |
|---|---|---|---|
| video-independent稳定任务轴 | v5.2/v6 | `Q_text` | 只作identity/QK，不独立写LoRA |
| task-grounded真实patch value | v5.2/v6 | `G_f` | value只能来自`P_f` |
| permutation-invariant高层Core | v5.2/v6 | mean backbone + centered residual + task-token blocks | 无frame-order、无Core-only factor路径 |
| source policy动作先验 | v4原始思想、v5.2/v6 | 8-probe `A_f→P_A` | 只参与Teacher–Policy gap，不独立授权LoRA |
| 视频动态与顺序信息 | v5.2/v6 | semantic/raw-visual relations→`P_T` | content与confidence分离，arm内重算 |
| 高效因果时序 | v5.2/v6 | 两层shared axial causal operator | Teacher/Policy values隔离，避免dense `16F` |
| slot-specific完整adapter编译 | v5.2/v6 | 320 routing slots | 先分别读Teacher/Policy，再编译gap |
| 足够的下游表达能力 | v6 | width256 factor hidden与八个full-width heads | 不从已证明够用的decoder挪走容量 |
| functional identity | v5.2/v6 | deterministic template + zero-init factor | 再增加`q·g`的连续identity合同 |
| 硬件友好规格 | v6 | width256、8×32、两层主干 | 用axial计算保留全部8+8 evidence |
| 无效视频尽量回到base | v5.2五臂 | teacher confidence + gap scale | wrong靠task relevance，乱序靠relation/temporal coherence回落 |

其中task-complete fast-decay是v6达到`143/400`的重要**训练**优势，不伪装成
模型模块。Loom首轮保持该recipe是为了继承已知可用的优化条件并隔离架构贡献；
它是否仍引起多task漂移，属于独立训练实验轴。

因此Loom对最强旧架构采取的是：

```text
保留已被正证据支持的访问、表示、时序、编译和identity能力
+ 为每种能力指定单一职责
+ 切断v7/v8/v10暴露的旁路和瓶颈
```

而不是退回旧拓扑，也不是为了“创新”删除有效能力。

## 15. 仍然需要实验验证的科学假设

复审没有发现新的结构矛盾，但以下语义不能仅靠公式宣称成立：

1. `P_T/P_A`经共享temporal operator和`W_T/W_A`后能否形成有意义的高层共同
   Procedure space；
2. 高imitation similarity是否确实对应较小必要LoRA，而非表示坍缩；
3. teacher confidence能否对correct保持高、对wrong/shuffled/reversed降低；
4. raw patch correspondence是否捕获robot/object co-motion而非背景；
5. gap结构是否减少same-task示范方差与多task能力漂移；
6. axial computation在最长视频上能否保持B20，或需真实profile后降B16。

这些是必须由probe、反事实和rollout验证的科学假设，不是需要继续添加结构才能
闭合的逻辑缺口。

## 16. 参数与效率合同

固定硬件友好规格：

```text
common width           256
attention heads        8 × 32
Core depth             2
shared axial depth     2
slot depth             1
FFN width              1024
factor hidden          256
Action probes          8
Teacher Events         8 per interval
```

相对旧Loom草案新增Text Meta、稳定patch readout、独立Action read、共同空间
alignment与confidence heads，同时用共享axial temporal避免双倍大网络。
设计级总量预计约`12.8M–13.3M` trainable parameters；实现前不得伪造精确数，
必须以真实module enumeration为准。

这是相对corrected rank-128 Source-SFT `10,297,344`的软预算提升。新增容量全部
用于：

- 已由v5.2/v6证明有效的task-grounded语义访问；
- teacher-visible motion/effect；
- Teacher–Policy gap；
- 连续置信尺度。

不扩大已排除为首要瓶颈的factor decoder，不为凑参数使用奇怪hidden width、
低秩共享或额外adapter。

计算上必须真实profile：

- 最长105-frame视频；
- B20首选；
- 若OOM或连续不稳才降B16；
- 只使用项目authority允许的GPU4–7；
- 不沿用v10显存结论。

## 17. 实现与机制验证

### 17.1 最短结构验证

- Text/VL/Action Meta-LoRA shapes与gradient；
- 8-token suffix native IDs/Gaussian rows；
- patch grounding value owner；
- Core frame permutation；
- arm-local relation recomputation；
- forward/backward matcher与mutual consistency；
- relation/event zero contracts；
- Event confidence bounds；
- Teacher/Policy axial causal masks；
- shared weights、separate memories；
- 320 slots、76 tensors、rank16；
- `q=0→identity`；
- `d=0→identity`；
- factor final zero-init；
- frozen source base；
- exact-resume与longest-video profile。

### 17.2 必做内部反事实

```text
fixed teacher, vary Action
fixed Action, vary teacher
teacher Event zero, vary Action
fixed Core, vary Teacher/Policy
fixed Teacher/Policy, vary Core
correct/same/wrong/shuffled/reversed
```

至少报告：

- Core relative L2；
- semantic/grid/matched relation RMS；
- matcher entropy与mutual consistency；
- raw energy与bounded confidence；
- 3 backbone / 5 learned Event贡献；
- `P_T/P_A`差异；
- teacher confidence `q_s`；
- aligned teacher/policy cosine；
- gap RMS与final `r_s`；
- Core-assisted gap RMS；
- effective LoRA与policy-action差异；
- task breadth与checkpoint漂移。

关键判定不再是“Action差异必须小”，而是：

```text
Action改变不能在teacher confidence=0时生成LoRA；
Action只能通过imitation gap改变适配；
correct teacher应同时具有可信q与有用gap；
无效视频不得仅因大transition获得高r。
```

### 17.3 闭环判定

保持single-checkpoint、paired、无放回：

```text
correct
same-task-other-video
wrong-video
shuffled
reversed
```

统一硬门仍为：

```text
correct400 >= 150
correct >= corrected Source-SFT best + 30
same与correct同档
correct显著优于wrong/shuffled/reversed
多个tasks共同贡献
独立RNG/video permutation复测成立
```

同时要求：

- high-match correct examples不被过度改写；
- low-match coherent correct examples获得更大但稳定的adapter；
- wrong因task relevance低而回落；
- shuffled/reversed因coherence/direction低而回落；
- gap从内部一直传到effective LoRA和policy action。

首轮仍使用normal-order positive-only AS，不加入order/contrast/margin negative
supervision。为隔离架构贡献，第一轮可保持当前task-complete训练合同；训练方法
优化是后续独立实验轴，不属于Loom结构定义。

## 18. 采用与生命周期

历史结论：

```text
EMBER Loom
= 逻辑闭合、可实现、可证伪的最终候选设计

current canonical
= Recenter
```

Loom当时的采用步骤为：

1. 先读本文与配套推导纪要；
2. 原位替换canonical Writer，不保留并行版本；
3. 使用fresh incompatible config/checkpoint schema；
4. 不加载v10 Writer weights；
5. 完成最短shape/gradient/identity/causal/resume vertical path；
6. 做最长视频真实profile；
7. 从functional identity fresh训练；
8. 按correct曲线、五臂与内部gap传递共同判定。

在主进程明确采用前，本文只承担设计交接，不修改活动代码、配置、训练或实验
状态。
