# Layer-Matched Memory Program Compiler

状态：2026-08-18 **active LMMPC-v5 implementation authority**。LMMPC-v1/v2/v3/v4均已完成macro25/50 strict与逐接口
诊断，terminal checkpoints只作证据，不再resume。owner已授权在EMBER稳定科学合同和本文主架构内完成局部迭代、
fresh训练、strict评测与逐接口分析；不得因尚未观察到性能峰值而过早终止，也不得在没有架构级证据时大幅改换
路线。

## 1. 一句话决策

LMMPC保留V6已经证明有效的task-grounded Semantic Core和Action-query Visual-Transition Procedure；同一次真实
π0.5 image/language/Action forward再产生逐层、逐rank的one-way memory states。V6 Procedure不再经过一组后置
routing identities恢复参数地址，而是直接读取每个`layer × rank`的memory时间序列；帧到video、video到K-set的
两级聚合都严格保留layer/rank轴。K-set的learned consensus只能作为per-video mean的逐cell有界修正；聚合后的
memory tensor本身进入同样受限的SHINE式group/rank axial M2P，再由共同训练的native factor heads生成一套完整
38-target rank16 A/B LoRA。

完整职责链是：

```text
language grounds visual evidence
    -> Semantic Core: objects / relations / goal

Action representation queries visual transitions
    -> Causal Procedure: what stage and how the task progresses

each layer/rank address uses all Procedure stages to read its memory sequence
    -> parameter-aligned per-video task memory

per-video order-preserving reduction
    -> permutation-invariant mean-anchored bounded K-set consensus
    -> Procedure-gated Core fusion
    -> axial M2P on the same memory grid
    -> native rank16 A/B
```

本文明确删除两个讨论阶段的错误接口：

- 不建立独立的320-slot bank或第二套parameter queries；数值上的`20×16=320`只是聚合memory tensor加两个边界行；
- 不把Action state和memory state任意相加为一个共同视觉Query。Action先形成高层Procedure，Procedure再读取memory。

v2只修正v1已经定位的`Procedure -> layer/rank memory reader`，并证明完整阶段确实进入parameter memory；但其
macro25/50 strict仅`71→73`，且训练后两层unbounded M2P把已经分离的Core-fused task Program重新变成共同方向。
v3把M2P改为逐cell有界residual，macro25恢复到`102`；但继续训练到macro50反而降到`60`。新的最早断点位于更上游：
K-set nonlinear correction相对per-video mean改写`10.19x→5.83x`，把between-task cosine在macro25/50分别从
`.6543/.7672`推到`.9025/.9218`。v4只把这一K-set correction改为mean-anchored逐cell bounded commitment，成功
关闭该覆盖接口，但macro25/50 strict仅`104→102`且churn52。对齐V6后，新的最早断点是
`Procedure -> layer/rank memory`：v4 reader把已有correct/reverse差异继续衰减，而V6的Core-conditioned slot reader
会放大有向差异。v5因此只把reader Query改为`policy address + task Core slot`，并在Procedure Keys上恢复RoPE；
memory-token动态仍是唯一Value。动态K、Core fusion、两个bounded commitment、native rank16 A/B与B20 functional
合同均不改。

## 2. 科学假设与历史继承

本轮检验的主假设是：

> V6已经能从语言、视频和source policy Action表征形成有用任务理解；其最早可改进接口是全局Procedure到具体
> policy layer/rank参数的承诺。若让Procedure直接读取真实layer/rank memory，并在地址保持的T/K聚合后共同训练
> native compiler，就可能同时保留V6 absolute、增强视频过程到LoRA的对应并避免冻结tail衰减。

| 历史证据 | LMMPC保留 | LMMPC删除或替换 |
| --- | --- | --- |
| v5/v5.2 | Core负责“是什么”，Procedure负责“怎样推进” | 无职责的单一video embedding |
| v6-fast143 | task-grounded视觉、Action查询transition、causal Procedure、rank16 A/B ownership | 全局Procedure经后置routing slots恢复layer/rank |
| LPCP143 | 同一次真实context中的layerwise policy evidence | 只把evidence作为冻结Procedure Query小修 |
| Dynamic-K/K4 | 每video先独立保序，K轴只聚合高层Program | frames/raw features/final LoRA平均或挑video |
| GOMQ151→135→131 | learned one-way memory有真实增益 | 37-token flat payload、rank32 B-only第二bank、首轮reward |
| SJNV/SFMC | hidden→factor必须逐stage检查 | 冻结W2把共同hidden方向旋散 |
| Direct Native | native A/B heads能打开BA/action写出 | 无结构flat B tail |
| SHINE/Doc-to-LoRA | layer-memory tensor自身进入结构化M2P | memory后再创建第二套参数地址 |
| v4 | 原生context必须有真实图文和Action prefix | trainable VL Meta旁路、absolute-time/action-phase shortcut |
| LMMPC-v1 | memory、动态K、M2P和rank16链路均可训练，macro25/50为`81→101` | 返回单个last-query再加独立endpoint、correct-minus-reverse和matching shortcut |
| LMMPC-v2 | 完整Procedure stage reader接通；macro25/50为`71→73` | unbounded M2P逐层放大并重写已形成的task/order signal |
| LMMPC-v3 | bounded M2P令macro25恢复到`102`，且compiler不再大幅覆盖Core-fused grid | K-set nonlinear correction无界覆盖per-video mean；继续训练到macro50降至`60` |

这只说明设计针对了已知失败，不能替代实验。尚未解决的风险仍包括：memory时间变化可能偏向低层nuisance；K-set
可能稳定错误task mean；axial M2P/factor heads可能再次衰减Program；shared functional credit仍可能造成task换手。

## 3. 完整部署数据流

```text
exact language L
+ K same-task action-hidden ordered videos V_k, stride 5
  (first claim: K in 1..4; training covers K1/K2/K3/K4)

for each video k independently:
  for each sampled frame t:
    native image I[k,t] + exact language prefix
      -> frozen VLM task-grounded evidence E[k,t,j]

    same native prefix
    + 50 fixed noise/time Action probes
    + 16 one-way learned rank-memory queries
      -> one Action-Expert context computation
      -> Action representation A[k,t]
      -> layer memory M[k,t,l,r], l=1..18, r=1..16

  E over frames
    -> V6 order-invariant Semantic Core C[k]

  adjacent task-grounded visual transition D[k,t]
  queried by A[k,t]
    -> V6 causal Procedure sequence P[k,1:T]

  P[k,1:T] reads M[k,1:T,l,r] at every fixed (l,r)
    -> directed per-video parameter memory H[k,l,r]

K videos:
  {C[k]} -> shared Semantic Core C_set
  {P[k]} -> shared Procedure summary P_set
  {H[k,l,r]} -> permutation-invariant mean-anchored bounded consensus at each fixed (l,r)
    -> H_set[l,r]

H_set dynamically opens the relevant C_set content
  -> Y[l,r], shape 18 x 16 x d

layer0/layer17 boundary projections
  -> Y_group[p,r], p in {action-in, layers1..18, action-out}
  -> shape 20 x 16 x d

zero-preserving group/rank axial M2P on this same tensor
  -> eight jointly trained native factor-family heads
  -> one complete 38-target rank16 LoRA

frozen source policy
  -> unseen-initialization closed-loop rollout
```

Writer在rollout前只运行一次。teacher action、proprio/state、reward、terminal、task ID、filename、object pose、hidden
normalization和policy outcome均不进入部署Writer。

## 4. 每帧只做一次有意义的native context计算

### 4.1 Task-grounded视觉证据

沿用V6的有效语义接口。exact language形成text task queries；每帧图像和语言经过冻结VLM，task queries读取真实
image patches并形成`E[t,j]`。`j`沿language/task-token轴保存对象、属性、关系和目标证据。

fresh LMMPC不新增trainable VL Meta-LoRA。冻结VLM表征后只训练Writer-local投影和readout，避免重新打开v4的
raw-image/VL-Meta bypass。

### 4.2 Action representation

每帧在真实image/language prefix后加入50个固定noise/time Action probes。Action Expert使用一套zero-init、
Writer-local Action Meta-LoRA形成probe states；source policy权重始终冻结。沿用V6最强、最清晰的readout：对最终
Action层的50个probe states作有效位置均值并投影得到`A[t]`。

`A[t]`不是teacher action，也不直接生成LoRA。它只表示source policy在当前画面和任务语言下对“此刻怎样交互”的
latent action hypothesis。

### 4.3 Layer/rank memory

同一native context中加入16个learned rank-memory queries。one-way mask保证native prefix和50个Action states看不到
memory；memory stream可读取完整prefix/Action K/V以及同层其它rank memories。每层收集：

```text
M[t,l,r] in R^1024, l=1..18, r=1..16
```

16对应当前生成LoRA的rank坐标，不是视频阶段数，也不为输出参数元素提供flat payload。第`l,r`个state只提供
“第l层、第r个rank坐标下的native contextual evidence”，它仍需Procedure解释和M2P解码，不能直接当LoRA。

内容forward每帧只执行一次；memory stream复用已捕获K/V。activation checkpoint只允许重算memory stream，不重复
image/language/Action内容forward。

## 5. 保留V6的任务理解前端

### 5.1 Semantic Core

对每条video独立构造V6式task-grounded Core：

```text
M_sem[j] = Mean_t(E[t,j])
centered[t,j] = E[t,j] - M_sem[j]
text query读取centered frame evidence
mean backbone + centered residual
  -> C_video[j]
```

它对该video的frame permutation严格不变，表达对象、关系、位置和目标；不承担动作顺序。K条video的Core只在各自
形成后做permutation-invariant set readout，得到`C_set`。correct/reverse/shuffle应共享近似相同Core。

### 5.2 Causal Procedure

沿当前输入顺序重算task-grounded视觉变化：

```text
D[0,j] = 0
D[t,j] = E[t,j] - E[t-1,j]
R[t] = Attention(Q=A[t], K=D[t,:], V=D[t,:])
Z[t] = A[t] + R[t]
P[1:T] = CausalTemporal(Z[1:T])
```

Procedure表达接近、接触、抓取、携带、接近目标、释放/建立关系等阶段。shuffle/reverse必须先重排真实frames，再
重算`E`顺序、`D`和causal positions；禁止重排已由correct顺序算好的transition。

Action的职责止于“把视觉变化解释为policy-relevant任务过程”。它不直接连接factor heads，也不与memory随意相加。

## 6. Procedure按阶段读取layer/rank memory

旧V6让固定policy地址读取完整、centered、有序Procedure。LMMPC-v1原本声称逐阶段读取memory，但实现实际返回
`attended[:, -1]`，所以只有`P_last`直接参与readout；随后又把不经过Procedure的`R_m[T]-R_m[0]`独立相加。
macro25/50 held诊断中attention只占最终reader输出的`2.75%/2.46%`，endpoint是attention的`41.0x/45.7x`；把整条
Procedure替换成重复`P_last`，输出逐元素不变。该接口必须原位替换。

v2先把每个固定地址的native memory变成只含动态内容的Value：

```text
R_m[t,l,r] = W_memory(Norm(M[t,l,r]))
X_m[t,l,r] = R_m[t,l,r] - R_m[0,l,r]
V_m[t,l,r] = X_m[t,l,r] - Mean_s(X_m[s,l,r])
```

对每个固定`(l,r)`只产生一个地址查询，让它读取完整Procedure阶段轴：

```text
S = P[last_valid]
Q[l,r] = Wq(Norm(S)) + W_address(Norm(E_layer[l] + E_rank[r]))
K[t] = Wk(Norm(P[t]))

H_video[l,r] = Wout Attention(
    Q[l,r],
    K[1:T],
    V_m[1:T,l,r]
)
```

职责是明确分开的：`P[t]`作为Key描述每一阶段是什么，固定layer/rank地址加task-level `S`决定该参数位置要读取哪些
阶段，同地址的centered native memory是唯一Value。`P[t]`已经由带真实frame position的causal Procedure产生，reader
不再添加第二个absolute-time clock。若内部Procedure退化为同一个向量，attention退化为均匀权重，而centered Value
按构造相消；因此模型不能只靠`P_last`或独立endpoint旁路阶段变化。constant video从首帧相对化后严格为零。

训练和部署都只运行正确正序一次：

```text
H_video(V) = StageRead(P(V), M(V))
```

不再内部构造`0.5 * (correct - reverse)`，也不硬编码`H(reverse)=-H(correct)`。reverse/shuffle控制必须先真实重排
raw frames，再完整运行同一个Writer；差异只能来自重算后的视觉变化、causal Procedure与阶段对应。

## 7. 两级聚合严格保留layer/rank地址

### 7.1 Frame到video

第6节对每个固定`(l,r)`独立处理时间轴。实现上：

```text
[B,K,T,L,R,D]
  -> [B*K*L*R,T,D]
  -> shared temporal/cross-attention
  -> [B,K,L,R,D]
```

不在此阶段对`l`或`r`求和、排序或做跨轴attention。因此第`l,r`个video memory按构造仍对应第`l,r`个LoRA地址。

### 7.2 Video到K-set

每条video独立得到：

```text
C_video[k]          semantic Core
P_summary[k]        high-level Procedure summary
H_video[k,l,r]      positive-order, stage-addressed parameter memory
```

先由DeepSets形式得到`C_set`和`P_set`。再对每个固定`(l,r)`：

```text
mu[l,r] = Mean_k(H_video[k,l,r])
centered[k,l,r] = H_video[k,l,r] - mu[l,r]
context[k] = concat(C_video[k], P_summary[k])

correction[l,r] = Mean_k(phi(centered[k,l,r], context[k], C_set, P_set))
Z_set[l,r] = mu[l,r] + psi(correction[l,r], C_set, P_set)

delta_set = Z_set - mu
limited_set = delta_set * min(1, RMS(mu) / max(RMS(delta_set), 1e-6))
gate_set = 0.5 * sigmoid(g_set)       # fresh g_set=0, initial gate=.25
H_set = mu + gate_set * limited_set
```

`phi/psi`共享、zero-bias；K=1时直接返回`mu=H_video`。K>1时learned set branch仍可比较不同video并提取共同
correction，但每个固定`(l,r)` cell上的实际修正始终不超过anchor `mu` RMS的`.5x`。mean只发生在已经保序理解、
且已映射到相同layer/rank坐标的高层memory，不平均frames、raw features或LoRAs。所有videos等权进入mean和
correction，不挑一条。

v3没有最后四行的commitment。macro25/50中raw `Z_set`相对`mu`的global relative-L2分别为`10.188/5.831`，
cosine仅`.245/.459`；between-task cosine由`mu`的`.654/.767`升到`.903/.922`，同时within-task condition cosine
由`.995/.997`降到`.965/.980`。因此该branch不是在稳定common Program，而是在覆盖地址保持的per-video mean。
只读`.25x` counterfactual把macro25/50的post-M2P between-task cosine从`.677/.692`降到`.521/.623`，并把
within-task cosine从`.983/.995`升到`.995/.997`；它会降低部分correct/reverse幅度，所以v4保留learned correction
并fresh共同训练，而不是删除branch或把counterfactual当成闭环成绩。

实现reshape为：

```text
[B,K,L,R,D]
  -> [B*L*R,K,D]
  -> permutation-invariant set block
  -> [B,L,R,D]
```

因此第二次聚合也不丢layer/rank correspondence。跨layer/rank通信只允许在所有T/K聚合完成后的M2P发生。

## 8. Procedure-gated Core融合

V6的高分依赖强Semantic Core，因此LMMPC不把Core降级成只改变attention权重；但Core也不能脱离视频动态独立写
LoRA。对每个已有`H_set[l,r]`读取相关Core内容：

```text
C_addr[l,r] = CrossAttention(
    Q = Norm(H_set[l,r] + E_layer[l] + E_rank[r]),
    K = C_set,
    V = C_set,
)

language_gate = tanh(W_language(Language_summary))
Y[l,r] = language_gate * (
    H_set[l,r]
    + tanh(W_dynamic(H_set[l,r])) * W_core(C_addr[l,r])
)
```

全部映射zero-bias。若动态video memory为0，则`H_set=0`且Core注入门为0；若language被移除，language gate为0。
因此：

- Core真实提供对象、关系和目标内容，继承V6 semantic backbone；
- Procedure/memory动态路径是Core进入LoRA的必要门；
- language-only、static Core、first-frame或video-presence不能独立写LoRA；
- correct/reverse/shuffle的差异必须来自Procedure/memory时序，而非Core身份。

## 9. Mean-anchored memory tensor直接进入bounded axial M2P

`Y[18,16,d]`已经是parameter-aligned memory grid，不再建立独立320 slots。两个边界行由现有grid产生：

```text
Y_in[r] = W_in(Y[layer0,r])
Y_out[r] = W_out(Y[layer17,r])

Y_group = concat(Y_in, Y_layers1..18, Y_out)
shape = [20 parameter groups, 16 ranks, d]
```

`20×16=320`只描述tensor cells，不是320个输入tokens、routing identities或第二套memory。

M2P proposal交替执行：

1. 固定rank，沿20个parameter groups做bidirectional group attention；
2. 固定parameter group，沿16个rank coordinates做bidirectional rank attention；
3. 重复少量blocks，并保持每个输出cell的group/rank index。

group/rank position、Core route和Procedure summary只进入Q/K与gate；Value只来自动态`Y_group`。attention和FFN均
zero-bias、zero-preserving，输入全零时proposal严格为零。v2直接把两层proposal作为factor输入；macro50实测第一层
相对anchor改写`4.500x`、第二层再改写`1.753x`，Core-fused的between-task cosine由`.3381`升到`.6560`，
correct/reverse relative-L2由`.2573`降到`.0938`。

v3保留相同两层axial proposal `Z`，但只允许它对每个固定`(group,rank)` cell作bounded refinement：

```text
delta = Z - Y_group
delta_limited = delta * min(1, RMS(Y_group) / max(RMS(delta), 1e-6))
gate = 0.5 * sigmoid(g)              # g fresh zero, initial gate = 0.25
Y_committed = Y_group + gate * delta_limited
Y_compiled = RMSNorm(Y_committed)
```

因此每个地址上的M2P correction始终不超过anchor RMS的`0.5x`，不能再覆盖该地址已经形成的视频动态Program；
axial blocks仍能学习跨层、跨rank协调，最终RMSNorm继续提供factor-head稳定尺度。对v2 macro50 hidden作只读
counterfactual，initial/max gate分别保持correct/reverse relative-L2=`.2479/.2308`、between-task cosine=
`.3608/.4056`，而same-task不同K4 set cosine仍为`.9928/.9939`。这只验证结构针对断点，不替代fresh闭环结果。

## 10. Native rank16 A/B生成

最终20×16 cells按V6验证过的native ownership进入八个共同训练、bias-free factor heads：

```text
q_A, q_B, v_A, v_B,
action_in_A, action_in_B,
action_out_A, action_out_B
```

第`l,r`个layer cell分别生成该层q/v LoRA的第`r`行A和第`r`列B；两个boundary rows生成action-in/out。按38-target
schema组装：

```text
A = A0 + deltaA(Y_group)
B = deltaB(Y_group)            # B0 = 0
DeltaW = B @ A
```

`A0`正常小随机、`B0`exact zero，constant/no-video保持source identity。heads沿用V6的family ownership和输出宽度，
但与LMMPC共同fresh训练，不加载冻结W2，不生成rank32第二bank，不使用fixed-A rank8子空间。

## 11. 训练合同

### 11.1 Fresh边界

LMMPC-v4改变了K-set commitment，正式checkpoint与v3/v2及V6/LPCP/GOMQ均不兼容。正式train24必须从fresh Writer
initialization开始，只复用冻结source policy、固定数据、normalization和公共LoRA schema。

开发bring-up可临时复用sealed V6 Core/Procedure activations检查memory→M2P→factor接线，但不训练成正式bridge、
不做闭环选模，也不把旧task能力算入新架构。

### 11.2 数据与K

- development只用固定train24产生梯度，每macro 24 tasks等权；
- video与action query同task跨episode；
- 每条video独立保序，K轴只在第7节发生；
- 训练真实覆盖K1/K2/K3/K4，连续完整宏周期内每task覆盖所有K；
- 多卡按K、总frames和历史cost均衡负载，但不改变task权重。

### 11.3 Objective

v2/v3首轮只使用**correct-order dense functional B20**：correct condition生成的LoRA作用于冻结source policy，在同task
跨episode action queries上下降functional loss。它直接回答生成方向是否对source policy有用，不混入reward。

v1的language/directed-Program matching从macro25的`.02114`降到macro50的`.001378`，同时Procedure cross-task cosine
却从`.7899`恶化到`.9503`；该loss没有cross-task wrong约束，并与pure-odd endpoint路径共同奖励“所有任务都学会正序
符号”。所以v2删除matching heads和reverse/shuffle训练臂，不把correct视频人为推成某个符号。same-task variance、
LoRA norm/rank/cosine、reverse/shuffle margin与expert reconstruction只作诊断；视频因果性最终由完整重前向的六臂
closed-loop裁决。

### 11.4 当前排除

首轮不加入Writer RL、生成LoRA后的task-local RL、expert bank、checkpoint union、gradient surgery、LoRA geometry、
rank/scale/seed sweep或额外target-task数据。若native BA/action已经material而held direction持续错误，reward/credit才
可作为后继的单一主要变量。

## 12. 实现与机制验证

实现必须原位替换terminal GOMQ运行面，保持一个canonical Writer：

- 复用现有one-way `backbone_memory` native context capture，37 capacity tokens原位替换为16 rank queries；
- 复用V6 semantic Core、Action-query transition和causal Procedure的已验证owner，删除旧后置slot compiler调用；
- 使用第6节stage-addressed centered-memory readout、地址保持且mean-anchored bounded的K-set、dynamic Core gate和
  group/rank axial M2P；
- 复用38-target schema、A0/B0 template和factor output shapes，新heads fresh；
- config/checkpoint/eval schema明确fresh-incompatible，不留legacy fallback；
- 训练/部署只走一次correct正序；评测reverse/shuffle重排raw frames后完整重跑Writer，不在模型内部复用correct结果；
- 最后一个frame microbatch用丢弃的zero rows补到固定shape后切掉padding，防止video排列或尾batch形状把正常BF16
  kernel差异放大成伪Procedure/memory方向；每个有效frame仍只forward一次，不使用batch1或重复有效frame；
- 不增加batch1、重复single forward、FP64训练、逐tensorhash或防御性扫描。

正式训练前必须通过：

1. 每帧内容forward计数为1，source policy参数0 gradient；
2. Action states不受memory影响，memory可读完整native context；
3. T/K aggregation不混layer/rank，K permutation只允许正常低位reduction差异；K>1的每cell set correction始终
   `<=.5x`对应per-video mean RMS，fresh gate和set branch均获得gradient；
4. no-language、no-video和constant路径的fresh effective BA为identity；
5. 用repeated-last替换Procedure内部阶段必须material改变H；correct/reverse/shuffle完整重前向后Procedure、memory、
   factor、BA/action均有material响应，且reverse不再是架构硬编码反号；
6. q/v/action-in/action-out八family均有finite nonzero gradient和native response；
7. longest-video K4无OOM/nonfinite，按真实samples/s选择batch；
8. full CPU suite与architecture guard通过，无平行旧runtime。

## 13. 充分训练、strict评测与性能峰值

smoke只证明接线。fresh formal首先训练到与历史强Writer有可比信息量的预注册节点，初始采用macro25或等价完整
task exposure，然后执行K4 strict paired400。该节点不是机械停止点：

- 若stage链健康、training和closed-loop仍有共同上升趋势，继续训练并评测相邻有意义checkpoint；
- 若首个结果一般但尚未形成峰值或平台，不因一次低分直接终止；
- 若Program→memory→factor/BA已明确数量级断裂，则不靠无限训练掩盖结构失败；
- 若约140+且retention/breadth改善，至少继续一个相邻节点；
- 首次约145或更高且retention合理，立即封存并补六臂，同时继续相邻checkpoint；
- 约145稳定、same-task鲁棒且视频controls合格，即使未超过150也可构成有价值结果；
- 任何单点高分若随后持续回落和换手，不能作为方法结果。

若好结果仍在积累，应训练到至少两个相邻有信息量checkpoint判断峰值和稳定性。若结果持续弱但未定位断点，先完成
逐stage和逐task分析，再在第15节允许范围内局部改进；不使用绝对分数门替代证据判断。

## 14. 全面分析合同

每个正式checkpoint至少分析：

```text
Semantic Core
V6 Causal Procedure
per-video Procedure-read memory H_video[l,r]
K-set memory H_set[l,r]
Core-fused grid Y[l,r]
post-M2P grid
native A/B and effective BA
fixed-action response
strict closed-loop
```

内部同时报告within-task跨video alignment、between-task separability、effective rank、train24/validation8 transport和
correct/reverse/shuffle差异，但只作定位。

strict paired400必须报告per-task、per-suite、breadth、相对V6-fast/LPCP143的retained/gained/lost、churn、Jaccard、
持续保留/失败/换手tasks。首次强checkpoint补：

```text
correct
same-task-other
cross-suite-wrong
shuffled
reversed
no-video
```

六臂严格配对state、env/policy RNG和video ordinal。正式选择只认single checkpoint；不以80-row screen、loss、union、
融合或挑task checkpoint选方法。

最早失败接口按下列口径定位：

- Core/Procedure不再保留V6顺序和语义证据：前端继承失败；
- Procedure有内容，H_video跨video近正交或低能：Procedure→memory readout失败；
- H_video健康，K-set压平task mean：多video共识失败；
- H_set健康，M2P/factor后断裂：compiler失败；
- BA/action material，closed-loop gained/lost不可分：functional credit/held occupancy失败；
- 早期高分后持续下降：shared retention/optimizer失败。

## 15. 允许的局部迭代与终局边界

owner要求本goal不自行大幅改换架构。允许的迭代必须保留以下主链：

```text
V6 Core/Procedure
-> Procedure reads layer/rank memory
-> address-preserving T/K aggregation
-> dynamic Core fusion
-> axial M2P
-> one native rank16 A/B LoRA
```

在明确最早断点后，可单变量调整：

- Procedure→memory的temporal readout、pool或directed/even channel分工；
- K-set consensus而不改成video/LoRA选择；
- Core的dynamic gate而不恢复language-only Value；
- axial M2P的最小通信职责或factor head开放方式；
- BA material后再调整functional/reward credit，但保持同一部署架构；
- optimizer/retention机制，但不靠挑checkpoint或多bank融合。

不得在本goal内无证据切换到expert dictionary、完全不同hypernetwork、第二套LoRA、rank/seed sweep或生成后task-local
RL。每次局部迭代都要充分训练、strict评测和逐接口分析，负结果只淘汰实际改动。

goal仅在以下之一成立时完成：

1. 同一shared single-checkpoint方法达到稳定约145+或更高，具有合理breadth/retention、相邻checkpoint稳定、
   same-task不同视频鲁棒且correct明显优于wrong/shuffle/reverse/no-video；或
2. 当前主链经过多轮充分训练、全面定位和有因果依据的局部改进仍无法达到资格，并形成重复、接口明确的终局证据。

## 16. 吞吐、GPU与正式运行

机制验证使用最小真实GPU；formal前按live状态同时检查gpu01/gpu02和`/data1`独立quota。单节点至多6张真正提高
吞吐的A40，有几张合适就用几张；低util、少量显存占用的设备在峰值余量足够时可与ycliu共驻。多卡固定
`NCCL_P2P_DISABLE=1`、GPU-local NUMA和deferred NCCL；独立evaluator使用persistent dynamic queue。

formal训练和评测必须来自clean pushed commit的detached frozen worktree，保留run contract、checkpoint、sampler/
RNG/topology、raw paired rows、aggregate、completion和必要analysis。接受正常BF16/TF32、batch和kernel低位差异，
以真实samples/s、LoRA/s、最长视频稳定性和closed-loop证据推进。

## 17. LMMPC-v1终局证据与v2 profile状态

LMMPC-v1同一fresh run的macro25 strict=`81/400`、breadth5、per-task=`2/0/32/3/0/39/5/0`；macro50 strict=
`101/400`、breadth5、per-task=`3/1/48/0/3/46/0/0`。25→50为`68 retained / 33 gained / 13 lost`、churn46，说明尚在
学习但能力明显换手；macro50相对LPCP143仅`83 retained / 18 gained / 60 lost`。本节以下profile仍是v1的历史运行
证据，不授权resume，也不封存v2。

clean pushed `4b6316a7ed5ba6e8cbe74e2e0bac377c11ed8e22`在gpu01物理`0/1/2/4/5/6`完成world6、
microbatch6、B20的两macro fresh profile：K1--K4每轮各6 tasks，macro=`32.97/29.83s`，peak allocated=
`40.52GB`，functional=`.15609→.15395`，Program matching=`.36194→.30113`，无OOM/nonfinite。

同一fresh macro2在task38、K4、323个有效frames的最长条件上通过机制门：每个有效frame只进入一次native context，
29个zero padding rows全部切除；correct/reverse parameter-memory relative-L2=`1.999976`，correct effective-BA
L2=`.564234`；constant从raw memory到compiled grid与effective BA均严格为0，K视频置换后的完整LoRA max-abs差严格
为0。该证据只封存运行recipe和机制健康，不是closed-loop性能证据。

### 17.1 Formal exposure correction and micro5 reseal

上述两macro profile没有覆盖100-macro schedule的真正最长条件。clean `de0b298` world5 formal在macro1--16的
functional/Program matching从`.15609/.36194`下降到`.11888/.17621`，但macro17的rank2在task38、K4、359帧条件
上未进入gradient all-reduce；单卡精确重放得到明确CUDA OOM，只差约96 MiB，因此其余rank的NCCL timeout是下游
表象，不是架构性能结论。

局部修正仅把functional B20 microbatch从6降为5。两者都执行4次policy forward，而micro5恰为`5×4`，不再为
`6+6+6+2`固定shape路径计算额外padding；Writer、输入帧、loss、task权重和架构均不变。micro5已通过原359帧
故障序列，并通过扫描完整100-macro schedule所得真实最大task38/K4/371帧条件，完整五任务序列峰值reserved=
`45,283,803,136` bytes。

clean pushed `dd81b94`随后在gpu02物理`1/2/3/4/7`以world5完成两macro full24 micro5 profile：macro=
`39.22/36.22s`，functional=`.15612→.15399`，Program matching=`.36194→.30113`，每轮K1--K4严格各6 tasks，
无OOM/nonfinite。部署路径同时省去只服务matching loss的shuffle Procedure/memory计算，单元合同证明其76个primary
LoRA tensors与训练路径逐元素相同。该profile与371帧证据共同重新封存formal recipe；原失败run无checkpoint，
不得resume，后继必须fresh。

### 17.2 Procedure-reader终局定位与v2门

macro25→50的cross-task cosine为：Core`.9219→.8486`、Procedure`.7899→.9503`、K-set`.7109→.7313`、compiled
`.6848→.7563`、final BA`.7489→.6431`。Core和final BA并非同步坍塌，最早异常正是Procedure及其memory承诺。

macro50 held validation8的reader counterfactual进一步显示：另一task的完整Procedure只让direct per-video H改变
`.125%`；zero Procedure虽然让shared stage改变`.501`，但主要是把后续K-set gate归零。reader component decomposition
显示独立endpoint占输出`.9968`，attention只占`.0246`；reverse endpoint逐元素严格为负。macro25已有同一结构，且
训练到macro50反而更依赖endpoint。因此旧trajectory即使继续提高loss/分数，也不能证明学会高层有向Procedure。

v2 config/checkpoint/eval schema全部fresh-incompatible。提交前worktree先行
通过结构和资源否决门：全量CPU=`284 passed`；world5两macro=`39.02/36.05s`、peak reserved=`34.20GB`；真实
task38/K4/323-frame中重复`P_last`或zero Procedure后parameter-memory仅剩正常norm的`.01390`且relative-L2约`1.0`；
reverse不再硬反号，constant identity、K置换、八family与reader梯度、source zero-grad均通过。shuffle在最终BA上仍只改
`.08864`，因此这里只证明结构能读取顺序，不能声称训练前已有正确时序优势。真实schedule最大371-frame及随后四任务
完整结束，peak allocated/reserved=`41,987,227,136 / 44,912,607,232` bytes。

随后clean pushed detached commit `61558f4550d732815f8e3f7b30504626e6deb577`原样复现全部seal门：world5两macro
用时=`39.08/35.97s`，functional=`.156120→.153996`、gradient norm=`.032812→.039320`，每轮K1--K4各6 tasks，
peak allocated/reserved=`31,714,849,792 / 31,889,293,312` bytes；clean 371-frame五任务序列peak为
`41,854,451,712 / 42,335,207,424` bytes。clean macro2机制结果与上述数值一致，且training/deployment recompile
LoRA max-abs为0。config现已`sealed`，允许从后继仅含seal authority的clean pushed commit启动fresh formal。

这些证据只封存结构、梯度、吞吐和资源合同，不是closed-loop或视频因果成绩；尤其shuffle BA差异`.08864`仍须由训练
后的strict controls裁决。

## 18. LMMPC-v2终局证据与v3可证伪门

v2 fresh world5在同一run完成macro25和exact-resume macro50。strict K4 correct分别为：

| checkpoint | correct | breadth | per-task | per-suite |
| --- | ---: | ---: | --- | --- |
| macro25 | `71/400` | 6 | `2/0/31/2/0/34/1/1` | `2/33/34/2` |
| macro50 | `73/400` | 6 | `1/0/35/13/5/15/4/0` | `1/48/20/4` |

25→50虽净增2，但只有`42 retained / 31 gained / 29 lost`，churn60、Jaccard`.4118`；Object净增15的同时Goal净丢14，
Goal6单task净丢19。loss在macro41--50约`.112`平台，closed-loop几乎不增而task能力继续换手。macro50相对同K4
LPCP143为`61 retained / 12 gained / 82 lost`、churn94；相对v1 macro50 101为`49/24/52`、churn76。故不续
v2 macro75，不补六臂，也不以两分增量声称尚在共同积累。

逐stage结果把否决边界限定在compiler而不是stage reader：macro50 Procedure correct/reverse relative-L2=`.8201`；
H_set=`.3980`；Core-fused grid=`.2573`，且同task不同K4 set cosine=`.9922`、between-task cosine=`.3381`。第一层
M2P已把order差异压到`.1502`并把between-task升到`.4943`；第二层进一步到`.0938/.6560`。最终effective BA的
correct/reverse relative-L2只剩`.0862`，different-K4 cosine=`.9998`。output RMSNorm单独几乎不改变这些指标，
所以最早破坏来自两层unbounded axial residual本身，而非Procedure趋同、K-set、factor量化或RMSNorm。

v3的首轮可证伪门是：

1. fresh机制门必须证明每cell correction在训练前后都满足`<=0.5x anchor RMS`，constant仍严格identity，VL
   Meta-LoRA参数与hook为0，八factor和M2P gate/blocks获得gradient；
2. validation8 stage门要求M2P后不再复现v2式task/order双重过度平滑；same-task正确K4鲁棒不能以correct/reverse
   近同为代价；
3. fresh训练仍到macro25并做strict400；若loss/closed-loop仍上升，继续macro50，不以首点判死；
4. 若v3已保持Program但closed-loop仍远低于v1/LPCP，最早断点后移到factor/functional credit，再在同一主链内
   做下一次局部迭代；不得回头恢复unbounded M2P/K-set或用rank/LR/seed小扫。

### 18.1 v3 clean mechanism/profile seal

clean pushed detached `987d131be3817f30afdb8513678a8daf9b1044e1`在gpu02物理`1/3/7`以world3完成两次
full24 macro：`58.55/55.32s`，functional=`.156120→.153991`，每轮K1--K4严格各6 tasks，profile peak
allocated/reserved=`35,437,871,616 / 35,720,790,016` bytes。另在物理7完整重放schedule world5/rank0的
macro44五任务序列，其中task38 K4为`82+105+92+92=371` frames；序列自然结束，peak allocated/reserved=
`41,851,758,080 / 42,393,927,680` bytes，无OOM/nonfinite。

同一macro2 checkpoint的task38真实K4机制探针显示：两层raw axial proposal相对Core-fused anchor的relative-L2高达
`32.2288`，证明v2式overwrite风险在fresh v3仍真实存在；bounded commitment实际只改写`.250003`，320个live
cells最大correction/anchor RMS=`.250395`，低于`.5`硬上限。gate=`.249998`，gate、两层M2P、memory/reader及八个
factor families梯度全非零，source policy非零gradient tensor数为0。

完整Procedure仍是必要路径：repeated-last使per-video parameter memory relative-L2=`.999716`、effective-BA=
`.493903`。reverse/shuffle在重新排列raw frames并完整forward后，compiled relative-L2=`1.06281/.36850`，effective-BA=
`.39081/.13498`；reverse不为架构硬反号。constant/template与K置换LoRA max-abs均为0，training/deployment
recompile max-abs为0。fresh checkpoint不包含任何VL Meta-LoRA tensor或runtime owner；相对v2恰删除921,600个冻结
参数并增加一个trainable scalar gate。

以上只通过结构、梯度、资源和顺序响应门，不证明correct视频沿有用closed-loop方向优于controls。v3 formal config由
这些clean artifacts封存；下一步必须fresh训练到macro25并做strict paired400，不能用机制距离代替性能裁决。

同一v3 runtime随后完成validation8×4 K4 fixed-panel生成profile：batch8/16/32的LoRA/s为
`.21314/.21489/.21627`，均覆盖226-frame最长样本且两次测量稳定；batch32 peak reserved=
`20,231,225,344` bytes、headroom=`27,468,496,896` bytes，零OOM/nonfinite/禁读。因此正式evaluation使用
batch32；这只封存部署吞吐，不参与科学选择。

## 19. LMMPC-v3终局证据与v4可证伪门

v3 fresh world3同一run完成macro25与exact-resume macro50；两次K4 strict paired400均完整exit0：

| checkpoint | correct | breadth | per-task | per-suite |
| --- | ---: | ---: | --- | --- |
| macro25 | `102/400` | 5 | `2/0/47/8/0/37/8/0` | `2/55/37/8` |
| macro50 | `60/400` | 6 | `2/0/24/2/1/26/5/0` | `2/26/27/5` |

25→50严格配对为`46 retained / 14 gained / 56 lost / 284 both-fail`，churn70、net`-42`、Jaccard`.396552`；
四suite只有Spatial净0，其余为`-29/-10/-3`。macro50相对LPCP143为`46/14/97`、churn111、net`-83`。因此
macro25不是仍在共同上升的早停点，而是随继续训练显著回落的单点；v3不得resume macro75或补六臂。

v3解决了v2的直接断点：macro25/50的M2P实际commitment相对Core-fused anchor仅`.24979/.24953`，而raw proposal
仍为`15.29x/12.83x`；Core-fused到compiled的correct/reverse只从`.3338→.3208`和`.2904→.2714`，不再出现v2
`.2573→.0938`的覆盖。gate保持`.2498/.2495`且链路持续有gradient。故不能把新回落重新归因于v3 bounded M2P
未生效。

新的最早断点是K-set nonlinear correction：

- macro25 per-video mean到raw K-set的relative-L2=`10.1880`、cosine=`.24493`、norm ratio=`10.3815x`；
- macro50仍为`5.83145/.45941/6.21550x`；
- between-task cosine在macro25由per-video mean`.65428`升到K-set`.90254`，macro50由`.76721`升到`.92178`；
- within-task condition cosine反而由`.99536/.99670`降到`.96486/.98036`；
- downstream Core fusion仍能把between-task降到`.64620/.65021`，bounded M2P仅小幅升到`.67674/.69152`，所以
  Core和M2P都不是更早的task-common来源。

只读mean-only与`.25/.5x` bound在两个checkpoint都恢复task分离和same-task coherence，但降低部分order/BA幅度。
这说明raw set branch同时含有有用有向成分，不能直接删除；也说明允许其覆盖mean不是提取cross-video common
Program的必要条件。v4因此只增加与M2P同形的逐cell mean-anchored bounded commitment，fresh gate初始`.25`、最大
`.5`；不加入matching、reverse arm、额外loss、rank变化、Core/Procedure变化或训练recipe变化。

v4首轮可证伪门：

1. fresh与训练后每cell K-set correction均`<=.5x mu RMS`，K1 exact identity、constant identity和K permutation不变；
2. set gate、phi/psi、reader、Core、M2P和八factor family均获得gradient，native BA/action material；
3. validation8 stage不再复现`mu -> H_set`的task-common overwrite，同时不要求correct/reverse距离人为增大；
4. fresh训练到macro25 strict400；若closed-loop与loss仍共同改善则继续macro50，若再次出现明确相邻回落则先定位；
5. v4若保持Program但absolute仍弱，下一变量必须后移到Core fusion/factor/functional credit，不能放开两个已证实的
   bounded commitment或用小参数sweep救结果。

### 19.1 v4 clean mechanism/profile seal

clean detached `8c40a56cb352ddd57098e646a10d3a2d32ec1c35`在gpu02物理`1/3/7`完成fresh world3两macro：
`58.13/54.83s`，functional=`.156120→.153994`，每轮K1--K4各6 tasks；peak allocated/reserved=
`35,550,641,664 / 37,971,034,112` bytes，零OOM/nonfinite/禁读。371-frame完整五任务序列自然完成，peak=
`41,987,913,216 / 44,920,995,840` bytes。

macro2 task38真实K4显示raw K-set proposal相对per-video mean仍为`6.40298x`，说明v3定位的overwrite压力在fresh v4
仍真实存在；mean-anchored commitment实际只改写`.250010x`，288个live cells最大correction/anchor RMS=
`.250531<.5`，gate=`.249997`。set gate、phi/psi、reader、Core、M2P和八factor families均获得gradient。M2P raw
proposal为`45.312x`，实际commitment同样只有`.250005x`。

repeated-last使parameter memory relative-L2=`.999913`、effective BA=`.933892`；reverse/shuffle完整重前向的BA
relative-L2=`.385271/.084918`。constant/template、K置换和deployment recompile max-abs均为0，source policy非零
gradient tensor为0，VL Meta-LoRA参数为0。

validation8×4 stage gate显示per-video mean→K-set的between-task cosine仅`.401466→.405721`、within-task cosine
`.990996→.979517`，不再复现v3 macro25/50的`.654/.767→.903/.922`覆盖；correct/reverse relative-L2由
`.491147→.516868`，Core-fused/compiled/effective BA仍为`.494697/.494893/.233755`。因此v4已经关闭预注册的最早
结构断点，但这不证明correct沿有用closed-loop方向；formal train24与strict400仍是裁决。

同一clean checkpoint在validation8×4固定long-first panel完成deployment generation profile。batch `8/16/32`
吞吐为`.212889/.214594/.216135 LoRA/s`，全部包含最长226-frame condition并稳定；batch32 peak reserved=
`20,231,225,344` bytes，仍有`27,468,496,896` bytes headroom，故正式评测选择batch32。三类禁读、OOM、nonfinite
均为0，Writer modules在rollout-scale handoff前释放。

## 20. LMMPC-v4终局证据与v5单变量设计

v4 fresh world4同一run完成macro25并按原world/topology exact-resume到macro50。两次K4 strict paired400均完整
exit0：

| checkpoint | correct | breadth | per-task | per-suite |
| --- | ---: | ---: | --- | --- |
| macro25 | `104/400` | 6 | `3/0/41/14/2/39/5/0` | `3/55/41/5` |
| macro50 | `102/400` | 6 | `1/0/39/4/0/46/10/2` | `1/43/46/12` |

25→50严格配对为`77 retained / 25 gained / 27 lost / 271 both-fail`，churn52、net`-2`、Jaccard
`.596899`。Object3净丢10，Goal6净增7，Long两task净增7；这不是稳定共同积累。macro50相对同schedule
LPCP143为`79/23/64`、churn87、net`-41`。top3 tasks占macro50成功的`.93137`，absolute、breadth和相邻稳定性
都不具资格；v4不得resume macro75或补六臂。

v4确实解决了v3的K-set覆盖。validation8×4中，per-video memory到bounded K-set的between-task cosine在
macro2/25/50仅为`.40147→.40572`、`.42747→.46325`、`.47445→.51123`；same-task K4 set cosine随训练为
`.97952→.99145→.99325`。Core-fused到bounded M2P的correct/reverse relative-L2在macro50几乎不变
`.15678→.15653`。因此不能重新归因于K-set或M2P未受约束。

LoRA也不是简单过小后停止学习。macro25→50 effective-BA norm从`27.285`增至`49.082`，两个checkpoint的
same-condition BA cosine仅`.61346`、relative-L2=`1.44061`；但strict不增。same-task四个K4 conditions的BA
centered variance仍约`.00125--.00579`，说明内部K4鲁棒没有阻止task换手。macro50仍低于LPCP的BA norm
`108.700`，但历史与本轮都已证明单纯增大norm不能选择closed-loop。

### 20.1 Procedure趋同不是单独根因

macro2→25→50中，raw Procedure的between-task cosine为`.89451→.95898→.96981`，centered temporal energy为
`.15489→.06561→.04748`；趋同确实存在。但同口径V6-LPCP macro25的raw Procedure between-task cosine更高，
达到`.99729`，仍获得`143/400`。因此不得用contrastive、matching或negative margin直接把raw cosine做低。

真正的分水岭在parameter readout：

| checkpoint/method | raw Procedure correct/reverse rel-L2 | parameter readout rel-L2 | readout/raw | compiled rel-L2 | BA rel-L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| v4 macro2 | `.72035` | `.49115` | `.682x` | `.49489` | `.23376` |
| v4 macro25 | `.49757` | `.22735` | `.457x` | `.22211` | `.17037` |
| v4 macro50 | `.43496` | `.18503` | `.425x` | `.15653` | `.14319` |
| V6-LPCP macro25 | `.10931` | `1.29212` | `11.82x` | `.26243` | `.26324` |

v4 reader用`P_last + layer/rank address`作Query，完整Procedure作Key，centered native memory作Value；task Core只在
readout之后进入fusion。训练使Query跟随同一Procedure endpoint，correct/reverse改变时Query与Key共同移动，最终
把有向差异自洽地抵消。V6则用固定policy routing加task Core作Query，Procedure只通过ordered Key/Value决定阶段，
所以即使raw Procedure高度趋同，slot readout仍能提取其小而有向的差异。v4的最早失效接口由此定位为
**Core-unconditioned Procedure endpoint Query -> parameter-addressed memory readout**，不是raw Procedure本身、K-set、
M2P、factor量化或LoRA强度。

### 20.2 v5唯一结构变量

v5保留v4每帧真实image/language/Action/memory context、V6 Core/Procedure、16个memory tokens、dynamic-K、两个
bounded commitments、20×16 axial M2P、native rank16 FactorHeads和全部训练/eval合同。只原位替换
`LayerRankMemoryReader`：

```text
address a[l,r] = RMSNorm(layer_id[l] + rank_id[r])

Core slot c[k,l,r]
  = CoreSlotReader(query=a[l,r], key=RMSNorm(C[k]), value=C[k])

Q[k,l,r] = Wq(a[l,r] + RMSNorm(c[k,l,r]))
K[k,t]   = Wk(RMSNorm(P[k,t])) with sampled-frame RoPE
V[k,t,l,r]
  = centered_t(Wm(M[k,t,l,r]) - Wm(M[k,first,l,r]))

H[k,l,r] = Wo Attention(Q[k,l,r], K[k,1:T], V[k,1:T,l,r])
```

这使四条流各守其职责：language/video Core决定“这个参数地址要找什么”，Procedure决定“在有向阶段轴上何时找”，
memory token提供“该policy layer/rank在真实context中的native Value”。每条video先独立形成同一18×16地址网格，
随后K轴仍只在相同地址逐cell聚合，因此两层聚合后不丢parameter correspondence。

Core只能改变attention Query，不能直接成为LoRA Value；constant video使`V=0`，无论language/Core是什么都必须输出
identity LoRA。v5不把Procedure直接加到Value，不新增language-only bypass，不引入correct-minus-reverse、matching、
negative arm或额外loss。若Core-conditioned Query仍不能让memory Value承诺有向过程，下一候选才是受限的centered
Procedure Value residual；它不属于本轮。

### 20.3 fresh边界与可证伪证据

reader参数和forward contract改变，v5 checkpoint/config/eval schema与v4不兼容，必须fresh；不得部分加载v4
Writer。实现只保留一个canonical runtime，不保留v4兼容分支。

1. 结构门：每帧仍只有一次native content forward；Core slot、Q/K、memory Value及八factor family都获得gradient；
   source policy zero-grad、constant identity、K permutation和training/deployment recompile合同不变。
2. 接口门：真实K4中替换Core必须改变parameter readout；重复`P_last`不能复现readout；correct/reverse不得是硬反号。
   相对matched v4 macro2，`raw Procedure -> H`的衰减必须有实质改善；若仍约`<=.6x`且没有其它有向stage增益，说明
   本轮变量没有击中接口，不启动昂贵formal。该门不要求模仿V6的`11.8x`，也不作为closed-loop替代指标。
3. 信息墙门：constant dynamic Value为零时，改变language/Core不能产生非零LoRA；teacher action/state/reward/
   terminal读数保持0。
4. 吞吐门：用真实最长K4与固定panel选择batch；不为低位浮点一致牺牲吞吐，不增加重复forward或防御性扫描。
5. 性能门：通过机制与资源门后fresh train24到macro25并做strict paired400；只要loss和closed-loop仍有真实共同上升
   证据就按同run exact-resume到macro50。最终以absolute、breadth、retained/gained/lost、churn和相邻checkpoint稳定性
   裁决，不以Procedure/BA距离选模。
