# Layer-Matched Memory Program Compiler

状态：2026-08-17 **active implementation authority**。简称 **LMMPC**。owner已授权在EMBER稳定科学合同和本文
架构思想内完成实现、fresh训练、strict评测、逐接口分析与局部迭代；不得因尚未观察到性能峰值而过早终止，也不得
在没有架构级证据时大幅改换路线。

## 1. 一句话决策

LMMPC保留V6已经证明有效的task-grounded Semantic Core和Action-query Visual-Transition Procedure；同一次真实
π0.5 image/language/Action forward再产生逐层、逐rank的one-way memory states。V6 Procedure不再经过一组后置
routing identities恢复参数地址，而是直接读取每个`layer × rank`的memory时间序列；帧到video、video到K-set的
两级聚合都严格保留layer/rank轴。聚合后的memory tensor本身进入SHINE式group/rank axial M2P，再由共同训练的
native factor heads生成一套完整38-target rank16 A/B LoRA。

完整职责链是：

```text
language grounds visual evidence
    -> Semantic Core: objects / relations / goal

Action representation queries visual transitions
    -> Causal Procedure: what stage and how the task progresses

Procedure reads layer/rank memory sequences
    -> parameter-aligned per-video task memory

per-video order-preserving reduction
    -> permutation-invariant K-set consensus
    -> Procedure-gated Core fusion
    -> axial M2P on the same memory grid
    -> native rank16 A/B
```

本文明确删除两个讨论阶段的错误接口：

- 不建立独立的320-slot bank或第二套parameter queries；数值上的`20×16=320`只是聚合memory tensor加两个边界行；
- 不把Action state和memory state任意相加为一个共同视觉Query。Action先形成高层Procedure，Procedure再读取memory。

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
  {H[k,l,r]} -> permutation-invariant consensus at each fixed (l,r)
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

## 6. Procedure读取layer/rank memory

旧V6先把Procedure压成全局表示，再让320个routing identities询问各参数位置。LMMPC改成先保留真实parameter
address，再由Procedure读取对应memory序列。

先共享投影memory并形成有向变化：

```text
R_m[t,l,r] = W_memory(M[t,l,r])
Delta_m[0,l,r] = 0
Delta_m[t,l,r] = R_m[t,l,r] - R_m[t-1,l,r]
Endpoint_m[l,r] = R_m[T,l,r] - R_m[0,l,r]
```

对每个固定`(l,r)`使用同一组共享参数：

```text
Q[t,l,r] = Wq(P[t]) + E_layer[l] + E_rank[r]
K[t,l,r] = Wk(R_m[t,l,r]) + E_time[t]
V[t,l,r] = Delta_m[t,l,r]

U[1:T,l,r] = CausalCrossAttention(Q[:,l,r], K[:,l,r], V[:,l,r])
F(video)[l,r] = U[last_valid,l,r] + W_endpoint(Endpoint_m[l,r])
```

这里Procedure是Query，因为它说明当前需要寻找哪个动作阶段；memory是K/V，因为它提供该layer/rank在各帧的native
状态及变化。这样继承V6的高层过程理解，又让memory真正成为LoRA生成的parameter-aligned内容，而非只改Query。

第一版保留一个显式directed channel，在同一组缓存`E/A/M`上重排并运行轻量V6 Procedure和memory readout：

```text
H_video(V) = 0.5 * (F(V) - F(reverse(V)))
```

它不重复VLM/Action backbone forward，只重复远小于backbone的transition、causal Procedure和memory readout。得到：

```text
H_video(reverse(V)) = -H_video(V)
```

反号约束只到parameter memory，不硬编码到LoRA。若held video上该directed channel近零并伴随绝对性能损失，说明
pure-odd readout删除了必要信息；允许在同一职责链内改为“directed additive channel + order-even gated semantic
channel”，但不得因此抛弃Core/Procedure→memory→M2P主架构。

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
H_video[k,l,r]      parameter-aligned directed memory
```

先由DeepSets形式得到`C_set`和`P_set`。再对每个固定`(l,r)`：

```text
mu[l,r] = Mean_k(H_video[k,l,r])
centered[k,l,r] = H_video[k,l,r] - mu[l,r]
context[k] = concat(C_video[k], P_summary[k])

correction[l,r] = Mean_k(phi(centered[k,l,r], context[k], C_set, P_set))
H_set[l,r] = mu[l,r] + psi(correction[l,r], C_set, P_set)
```

`phi/psi`共享、zero-bias；K=1时correction严格为0，`H_set=H_video`。mean只发生在已经保序理解、且已映射到相同
layer/rank坐标的高层memory，不平均frames、raw features或LoRAs。所有videos等权进入mean和correction，不挑一条。

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

## 9. Memory tensor直接进入axial M2P

`Y[18,16,d]`已经是parameter-aligned memory grid，不再建立独立320 slots。两个边界行由现有grid产生：

```text
Y_in[r] = W_in(Y[layer0,r])
Y_out[r] = W_out(Y[layer17,r])

Y_group = concat(Y_in, Y_layers1..18, Y_out)
shape = [20 parameter groups, 16 ranks, d]
```

`20×16=320`只描述tensor cells，不是320个输入tokens、routing identities或第二套memory。

M2P交替执行：

1. 固定rank，沿20个parameter groups做bidirectional group attention；
2. 固定parameter group，沿16个rank coordinates做bidirectional rank attention；
3. 重复少量blocks，并保持每个输出cell的group/rank index。

group/rank position、Core route和Procedure summary只进入Q/K与gate；Value只来自动态`Y_group`。attention和FFN均
zero-bias、zero-preserving，输入全零时输出严格为零。该结构允许深层memory反向帮助浅层LoRA并协调q/v/action，
同时不让一个全局MLP重新学习全部参数地址。

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

LMMPC改变了Procedure→memory、T/K聚合、Core融合和M2P/compiler，正式checkpoint与V6/LPCP/GOMQ均不兼容。正式
train24必须从fresh Writer initialization开始，只复用冻结source policy、固定数据、normalization和公共LoRA schema。

开发bring-up可临时复用sealed V6 Core/Procedure activations检查memory→M2P→factor接线，但不训练成正式bridge、
不做闭环选模，也不把旧task能力算入新架构。

### 11.2 数据与K

- development只用固定train24产生梯度，每macro 24 tasks等权；
- video与action query同task跨episode；
- 每条video独立保序，K轴只在第7节发生；
- 训练真实覆盖K1/K2/K3/K4，连续完整宏周期内每task覆盖所有K；
- 多卡按K、总frames和历史cost均衡负载，但不改变task权重。

### 11.3 Objective

首轮使用两类互补信号，不混入reward：

1. **dense functional B20**：correct condition生成的LoRA作用于冻结source policy，在同task跨episode action queries
   上下降functional loss；它提供policy-useful方向；
2. **language/directed-Program matching**：`P_summary`、`H_video`和`H_set`应匹配自己的exact language；same-task
   videos为正样本，shuffle/reverse为Program级order negatives。negative arms不读取action/reward/outcome，也不规定
   negative LoRA必须失败。

matching系数只做一次初始量级校准，使首个完整macro的标量贡献约为functional的10%，随后冻结；不做lambda sweep。
same-task variance、LoRA norm/rank/cosine和expert reconstruction只作诊断，不加入首轮loss。

### 11.4 当前排除

首轮不加入Writer RL、生成LoRA后的task-local RL、expert bank、checkpoint union、gradient surgery、LoRA geometry、
rank/scale/seed sweep或额外target-task数据。若native BA/action已经material而held direction持续错误，reward/credit才
可作为后继的单一主要变量。

## 12. 实现与机制验证

实现必须原位替换terminal GOMQ运行面，保持一个canonical Writer：

- 复用现有one-way `backbone_memory` native context capture，37 capacity tokens原位替换为16 rank queries；
- 复用V6 semantic Core、Action-query transition和causal Procedure的已验证owner，删除旧后置slot compiler调用；
- 新增Procedure→memory causal readout、地址保持的K-set、dynamic Core gate和group/rank axial M2P；
- 复用38-target schema、A0/B0 template和factor output shapes，新heads fresh；
- config/checkpoint/eval schema明确fresh-incompatible，不留legacy fallback；
- reverse/shuffle只重排缓存frame evidence并重跑轻量Procedure/memory readout，不重复backbone forward；
- 不增加batch1、重复single forward、FP64训练、逐tensorhash或防御性扫描。

正式训练前必须通过：

1. 每帧内容forward计数为1，source policy参数0 gradient；
2. Action states不受memory影响，memory可读完整native context；
3. T/K aggregation不混layer/rank，K permutation只允许正常低位reduction差异；
4. no-language、no-video和constant路径的fresh effective BA为identity；
5. correct/reverse/shuffle重新计算且Procedure、memory、factor、BA/action均有material响应；
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
