# Layer-Matched Memory Program Compiler

状态：2026-08-17 **active LMMPC-v2 implementation authority**。LMMPC-v1已完成macro25/50 strict与逐接口诊断，
其terminal checkpoint只作证据，不再resume。owner已授权在EMBER稳定科学合同和本文主架构内完成v2实现、fresh
训练、strict评测、逐接口分析与局部迭代；不得因尚未观察到性能峰值而过早终止，也不得在没有架构级证据时大幅
改换路线。

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

each layer/rank address uses all Procedure stages to read its memory sequence
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

v2只修正v1已经定位的最早断点：`Procedure -> layer/rank memory reader`。Core、Action/Procedure前端、one-way
memory、动态K、Core fusion、axial M2P、native rank16 A/B与B20 functional合同均不改。

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

v2首轮只使用**correct-order dense functional B20**：correct condition生成的LoRA作用于冻结source policy，在同task
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
- 使用第6节stage-addressed centered-memory readout、地址保持的K-set、dynamic Core gate和group/rank axial M2P；
- 复用38-target schema、A0/B0 template和factor output shapes，新heads fresh；
- config/checkpoint/eval schema明确fresh-incompatible，不留legacy fallback；
- 训练/部署只走一次correct正序；评测reverse/shuffle重排raw frames后完整重跑Writer，不在模型内部复用correct结果；
- 最后一个frame microbatch用丢弃的zero rows补到固定shape后切掉padding，防止video排列或尾batch形状把正常BF16
  kernel差异放大成伪Procedure/memory方向；每个有效frame仍只forward一次，不使用batch1或重复有效frame；
- 不增加batch1、重复single forward、FP64训练、逐tensorhash或防御性扫描。

正式训练前必须通过：

1. 每帧内容forward计数为1，source policy参数0 gradient；
2. Action states不受memory影响，memory可读完整native context；
3. T/K aggregation不混layer/rank，K permutation只允许正常低位reduction差异；
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

v2 config/checkpoint/eval schema全部fresh-incompatible，当前仍为`unsealed_pending_live_profile`。当前worktree已经先行
通过结构和资源否决门：全量CPU=`284 passed`；world5两macro=`39.02/36.05s`、peak reserved=`34.20GB`；真实
task38/K4/323-frame中重复`P_last`或zero Procedure后parameter-memory仅剩正常norm的`.01390`且relative-L2约`1.0`；
reverse不再硬反号，constant identity、K置换、八family与reader梯度、source zero-grad均通过。shuffle在最终BA上仍只改
`.08864`，因此这里只证明结构能读取顺序，不能声称训练前已有正确时序优势。真实schedule最大371-frame及随后四任务
完整结束，peak allocated/reserved=`41,987,227,136 / 44,912,607,232` bytes。

这些是提交前worktree smoke，不进入formal config的sealed evidence。下一步必须从clean pushed detached commit原样重跑
两macro与371-frame门，写回source commit/run root后再允许formal；不得把本节数值冒充closed-loop或视频因果成绩。
