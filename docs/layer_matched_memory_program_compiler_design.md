# EMBER-LMMPC Core-Addressed Reader

状态：2026-08-18 **sealed tested architecture，formal non-pass**。本文记录当前仓库实现所对应的完整架构、训练合同和
最终证据；它不是active design，不包含下一步方案，也不授权resume或恢复历史Writer。

历史revision v1--v4及其逐阶段裁决已收敛到`research_history.md`。当前代码/config identity仍保留历史后缀`v5`，
但方法名称统一为**EMBER-LMMPC Core-Addressed Reader**。

## 1. 被检验的架构假设

语言、视频、source-policy Action representation和layer/rank memory承担不同职责：

- language确定对象、关系、目标和task query；
- ordered video提供阶段变化与有向过程；
- native Action representation把视觉变化解释为policy-relevant过程；
- memory states把内容证据放入Action-Expert layer与LoRA rank对应的坐标。

Core-Addressed Reader让task Semantic Core查询完整有序Procedure，并从同一native context的layer/rank memory中读取
Value。两级T/K聚合保持参数地址，随后在同一grid上完成bounded M2P并生成一套完整rank16 LoRA。

这套假设从fresh Writer接受了完整formal检验；最终结果未达到性能与稳定性资格。

## 2. 输入、输出和信息墙

输入：

```text
exact task language
+ K same-task action-hidden ordered videos
K in {1,2,3,4}, frame stride 5
```

输出：

```text
one complete 38-target native rank16 A/B LoRA
```

部署Writer只运行一次。禁止输入teacher action、proprio/state、reward、terminal、task ID、filename、object pose、
hidden normalization和policy outcome。source policy全部冻结；没有第二套adapter、expert route、video LoRA平均或
checkpoint融合。

## 3. 完整数据流

```text
for each video independently:
  each sampled frame + exact language
    -> frozen VLM task-grounded evidence

  same native prefix
  + 50 fixed noise/time Action probes
  + 16 one-way learned rank-memory queries
    -> one native image/language/Action context computation
    -> Action representation A[t]
    -> M[t, 18 Action-Expert layers, 16 rank coordinates]

  frame evidence -> order-invariant Semantic Core C[k]

  adjacent task-grounded visual transitions queried by A[t]
    -> causal Procedure P[k, 1:T]

  for each fixed (layer, rank):
    Core-conditioned address Query
    + ordered Procedure Keys
    + centered native memory Values
      -> H_video[k, layer, rank]

K videos:
  C[k] -> shared Core
  H_video[k, layer, rank]
    -> same-address permutation-invariant bounded K-set
    -> H_set[layer, rank]

H_set gates relevant Core content
  -> 18 x 16 grid
  -> action-in/action-out boundary rows
  -> bounded 20 x 16 group/rank axial M2P
  -> eight native FactorHeads
  -> complete rank16 LoRA
```

## 4. 每帧native context

### 4.1 Task-grounded视觉证据

exact language形成task-token queries；冻结VLM产生每帧image-patch evidence。Writer-local grounding读取对象、属性、
关系和目标内容。当前架构没有trainable VL Meta-LoRA；VLM原生参数保持冻结。

### 4.2 Action representation

真实image/language prefix后追加50个固定noise/time Action probes。Action Expert使用Writer-local Action Meta-LoRA形成
probe states；最终Action层有效位置均值投影为`A[t]`。它不是teacher action，也不直接生成LoRA，只用于解释视觉
transition。

### 4.3 Layer/rank memory

同一context加入16个one-way memory queries。native prefix和Action probes看不到memory；memory stream可以读取同层
native context。18层各收集16个state：

```text
M[t, l, r] in R^1024, l=1..18, r=1..16
```

16代表LoRA rank坐标，不是video阶段数。每个有效frame只执行一次内容forward；memory replay复用native K/V。

## 5. Semantic Core和causal Procedure

每条video先独立形成两种表示：

```text
Core: task-grounded frame mean + task-selected centered visual evidence
Procedure: Action-query visual transition -> causal temporal encoder with frame positions
```

Core表达对象、关系与目标，对frame permutation近似不变。Procedure保留接近、抓取、移动、释放等阶段顺序。
shuffle/reverse控制必须先重排raw frames，再完整重算visual transition、positions和Writer forward。

## 6. Core-addressed layer/rank memory reader

对每个固定`(l,r)`地址：

```text
address[l,r] = Norm(layer_id[l] + rank_id[r])
core_slot[l,r] = CoreSlotReader(address[l,r], C_video)

Q[l,r] = Wq(address[l,r] + Norm(core_slot[l,r]))
K[t]   = Wk(Norm(P[t])) with sampled-frame RoPE

projected[t,l,r] = Wm(Norm(M[t,l,r]))
relative[t,l,r]  = projected[t,l,r] - projected[first,l,r]
V[t,l,r]         = relative[t,l,r] - Mean_t(relative[t,l,r])

H_video[l,r] = Wo Attention(Q[l,r], K[1:T], V[1:T,l,r])
```

language/video Core决定该参数地址寻找什么，Procedure决定沿有向阶段轴何时读取，memory提供该policy layer/rank的
native Value。Core不能直接成为LoRA Value；constant video使centered Value为零并保持identity。

## 7. 两级地址保持聚合

### 7.1 Frame到video

时间轴对每个`(layer,rank)`独立读取：

```text
[B,K,T,L,R,D] -> [B*K*L*R,T,D] -> [B,K,L,R,D]
```

此阶段不跨layer/rank混合。

### 7.2 Video到set

每个同地址cell先求per-video mean，再由共享DeepSets branch提出correction。最终correction以per-cell RMS限制：

```text
mu = Mean_k(H_video[k,l,r])
proposal = SetNetwork(H_video[:,l,r], Core, Procedure summary)
limited = Bound(proposal - mu, anchor_rms=RMS(mu))
H_set = mu + (0.5 * sigmoid(g_set)) * limited
```

K1严格返回`mu`。最大learned correction不超过anchor RMS的`.5x`。mean发生在每条video已经独立保序并对齐到相同
policy地址之后，不是frame、raw feature或LoRA平均。

## 8. Dynamic Core fusion和bounded M2P

H_set作为动态Query读取shared Core。Core Value同时受nonzero memory gate和language gate限制；no-video、constant
或no-language路径保持identity。

18个expert-layer rows增加action-in/action-out边界后形成`20 x 16 x 256`grid。数值上的320 cells不是另一套tokens
或routing bank。两层axial blocks分别沿20个parameter groups和16个rank coordinates通信，保留每个cell的地址。

raw axial proposal只作为bounded refinement：

```text
delta = proposal - addressed_grid
limited = Bound(delta, anchor_rms=RMS(addressed_grid))
committed = addressed_grid + (0.5 * sigmoid(g_m2p)) * limited
compiled = RMSNorm(committed)
```

每cell correction最大同样不超过anchor RMS的`.5x`。

## 9. Native rank16 LoRA生成

八个共同训练、bias-free FactorHeads：

```text
q_A, q_B, v_A, v_B,
action_in_A, action_in_B,
action_out_A, action_out_B
```

18层cells生成q/v factors，两个boundary rows生成action-in/out factors。38-target模板为：

```text
A = A0 + deltaA(compiled_grid)
B = deltaB(compiled_grid), B0 = 0
DeltaW = B @ A
```

Writer和FactorHeads从fresh初始化，未加载历史compiler或carrier。

## 10. 正式训练合同

- fixed development train24 only；
- 每macro 24 tasks等权；
- K1/K2/K3/K4每macro各6 tasks；
- video与action query同task跨episode；
- correct-order dense functional B20 only；
- frozen source policy，per-task B20 loss先均值再跨24 tasks等权；
- AdamW，BF16，peak LR `.0003`，100 macros；
- checkpoint macros `25/50/75/100`；
- exact-resume锁定world6/topology；
- 不含matching、reverse/shuffle loss、reward、expert reconstruction或LoRA geometry objective。

正式config：`configs/pi05_writer_layer_matched_memory_program_compiler_v5.json`。其中顶层launch-time status保持不可变；
当前non-pass状态由本文、`progress.md`和formal结果共同定义。

## 11. 机制与吞吐封存

clean commit`86c9e63cb5693de206c2baa27be999d55df771ed`通过：

- 每有效frame一次native context；source policy 0 gradient；
- full Procedure和Core Query都会material改变parameter memory；
- reverse不是架构硬反号；constant/template和K permutation delta为0；
- K-set、M2P、memory、reader和八factor families梯度非零；
- validation8 reader/raw correct-reverse relative-L2由旧reader约`.718x`提高到`1.819x`；
- H_set within-task cosine约`.970`、between-task约`.245`；
- longest K4 371-frame sequence无OOM/nonfinite；
- K4 deployment batch32约`.216 LoRA/s`，显存余量充分。

这些证据证明图接通和reader命中预注册接口，不证明closed-loop方向正确。

## 12. 正式closed-loop结果

同一fresh world6 run：

| macro | strict | breadth | per-task | per-suite |
| ---: | ---: | ---: | --- | --- |
| 25 | `123/400` | 8 | `3/3/44/25/1/43/3/1` | `6/69/44/4` |
| 50 | `84/400` | 5 | `0/1/45/1/0/29/8/0` | `1/46/29/8` |
| 75 | `89/400` | 6 | `3/0/36/1/2/44/3/0` | `3/37/46/3` |
| 100 | `87/400` | 4 | `0/4/38/0/0/42/3/0` | `4/38/42/3` |

相邻retained/gained/lost为`71/13/52`、`59/30/25`、`70/17/19`。400 rows只有49行始终成功；macro25到50丢失的
52行到macro100只恢复15行。最佳123仍比同schedule LPCP143净低20、比GOMQ151净低28，且91.1%成功集中于三个
tasks。

固定K4+B20 loss为`.112124 -> .099353 -> .098427 -> .101337`。25到50 loss显著改善而strict净丢39；Program与
FactorHeads cross-decode又表明二者都持续material变化。完整配对和归因见`architecture_reasoning.md`。

## 13. 终局证据边界

该formal recipe不通过：无checkpoint达到约145，峰值后没有恢复，多task breadth最终降到4，因此没有补六臂。

本实验确立：

- Core-addressed reader相对matched旧reader具有真实闭环正收益；
- Dynamic-K、layer/rank memory、bounded K-set/M2P和完整native rank16 LoRA可共同训练部署；
- 当前static cross-episode B20 recipe不能让一个shared checkpoint稳定积累held support；
- 同task跨video coherence、material Program和健康LoRA统计不足以推出closed-loop共同积累。

本实验没有确立：

- 当前Program已理解高层过程而非task identity；
- FactorHeads、functional occupancy、optimizer或它们的交互中哪一个是唯一根因；
- memory token一般、Dynamic-K一般或rank16一般无效；
- correct视频具备六臂因果优势。

当前没有active successor。所有未决归因保持开放，本文不包含推荐的解决方案。

## 14. Remote-visible provenance

- current implementation：`src/ember/writer/`；
- canonical launcher：`scripts/train_as_writer.py`；
- formal config：`configs/pi05_writer_layer_matched_memory_program_compiler_v5.json`；
- full historical ledger：`docs/research_history.md`；
- independent evidence synthesis：`docs/architecture_reasoning.md`。

本地formal roots（未提交大型artifact）：

- `runs/outputs/pi05_lmmpc_v5_formal_fresh_r6_b20_aecbce5_gpu01p124567_20260818`；
- `runs/analysis/lmmpc_four_checkpoint_strict_trajectory_20260818.json`；
- `runs/analysis/lmmpc_program_factorheads_cross_decode_macro25_50_75_100_20260818.json`；
- `runs/analysis/lmmpc_macro25_50_75_100_drift_diagnosis_20260818.json`。

外部review不应假定这些本地路径可访问；影响判断的数值已经重述在本文和`architecture_reasoning.md`。
