# Q/V Rank-Reserved Native Reward Compiler

状态：2026-08-11 active design authority。本文只授权先做现有 frozen-v6 macro0 与
Reward-Credit cycle1 Program 的 load-only 行为裁决；在这两个裁决完成前，不授权新训练。

## 1. Why this is the next single-variable intervention

Reward-Credit cycle1 已完成正式训练，但同一paired correct400仍为`134/400`、breadth6，
相对zero-Program macro0恰好`14 gained / 14 lost`。这不能证明reward credit无效，因为随后沿
部署链逐层定位得到：

1. correct视频的P256 Program motion具有task-common与same-task video一致性，wrong、shuffled、
   reversed输出量级约为correct的`1.15%`；
2. analytic FactorHead tangent及effective `BA`仍保留该结构，task-common/same-video cosine约
   `.98/.96`；
3. 首个失败接口是36个q/v target的原生BF16 factor materialization：连续factor delta约
   `1e-8 RMS`，而非零BF16 A/B的局部ULP约`1e-4`，own-target native cosine仅约`.037`；
4. action的四个factor为FP32，不存在同一sub-ULP问题。

因此下一实验只改变q/v的原生LoRA编译拓扑，不改变视频encoder、Balanced P256 key、Program、
Reward objective、source policy、split、rank预算、部署输入或action路径。它直接回答：已经学到的
视频条件Reward方向在可部署LoRA中可达后，是否能提高真实闭环性能。

## 2. Rejected numerical patches

以下候选已有直接证据，不能重新扫描：

- 把q/v cache从BF16改为FP16：相同2-byte storage下native cosine约`.064`，仍失败；不得为此
  改dtype或重复throughput profile。
- fixed dither、ULP/local coordinate descent、gauge、global scale：均未把stable target可靠落入
  原生格点；最好local-CD same-video cosine约`.217`且error-to-zero约`.977`。
- 对`G0`和`G0+T`分别做absolute rank16 refactorization：原生量化jitter淹没Reward motion，
  delta cosine仅`.0045--.0213`，norm ratio约`8.7--182`。
- balanced rank14/15 base factorization：base effective error约`.24%`，相对Reward motion过大。
- rank1 residual：只捕获约`.919` tangent energy；rank2已捕获`.9996`，rank4没有足以抵偿
  两个额外base slots的收益。
- 固定global carrier、被舍弃的旧A rows或task-expert basis：分别退化为历史Tangent-Basis、低捕获
  carrier或Expert-Manifold runtime dictionary，不是本设计。

## 3. Exact deployment graph

信息墙保持：

```text
exact task language + exactly one action-hidden teacher video
  -> frozen video/condition path
  -> P256 Program code
  -> frozen-v6 base factors F0 and analytic Reward tangent dF
  -> one complete 38-target public rank-16 LoRA
  -> frozen source policy
```

video仍是唯一dynamic value。不得增加language-only LoRA bypass、task ID、expert route、第二套LoRA、
multi-video、checkpoint融合、teacher action/proprio/reward/terminal或held oracle。

### 3.1 Stable analytic tangent

对当前FactorHead，必须先在连续空间计算稳定JVP，而不是先对微小factor delta做原生cast：

```text
z  = W1 S
dz = W1 R
dF = W2 ((GELU'(z) cast to the sealed decoder compute dtype) * dz)
```

`S`为frozen-v6 condition state，`R`为当前Program residual。对每个target把`dF`拆成`dA,dB`，q/v的
一阶effective tangent严格定义为

```text
T = B0 dA + dB A0
```

这里有意不含二阶`dB dA`：`T`是FactorHead在`F0`处的微分，不是把两个finite factor候选相乘后的差。
load-only cycle1必须读取现有正式checkpoint的非零Program；correct-video macro0使用exact zero Program。

### 3.2 Q/V: pivot-preserving rank14 base plus two physical zero-B slots

对36个q/v target逐condition处理原生v6 factors `B0[output,16]`、`A0[16,input]`：

1. 用固定、确定性的modified Gram-Schmidt column pivot规则从原生`B0`选择14列；tie-break必须由
   原始column ordinal决定，不能依赖task outcome。
2. `B_base=B0[:,keep14]`保持bit-exact，不重新factorize或混合gauge。
3. 在连续compute dtype中解
   `A_base = argmin_A ||B_base A - B0 A0||_F`，随后按public target原生dtype落盘。
4. public slots 14和15的base `B`必须为physical exact zero；base与residual slots之后不得做任何
   旋转或gauge mixing。
5. 对上述stable effective tangent `T=B0 dA+dB A0`取condition-local compact rank2 SVD，组装balanced factors
   `B_res=U2 sqrt(S2)`、`A_carrier=sqrt(S2) V2h`；这是同一video条件生成的carrier，不是global basis。
6. 最终单一LoRA为
   `B=[B_base | B_res]`、`A=[A_base; A_carrier]`，因此
   `BA=B_base A_base + B_res A_carrier`，不存在base/residual cross terms。

当correct-video的Program为零时，`B_res`必须exact zero。`A_carrier`可采用确定性finite占位值，但不得影响
effective `BA`；此时部署的是rank14 frozen-v6 base，并不是source-policy identity。真正no-video必须走显式
fast path，完全跳过pivot/lstsq/SVD并直接返回template-A/zero-B source identity。正式可训练版本另见第6节，
不能通过zero-matrix SVD backward构造。

### 3.3 Action: unchanged full-rank16 FP32 path

两个action targets保持原full-rank16 FP32 factor路径：实际候选为
`(B0+dB)(A0+dA)`，因此保留很小但真实存在的二阶`dB dA` cross term，不做rank14压缩或rank2近似。
原因是action没有BF16 sub-ULP问题，而action tangent的rank2近似仅有约`.91--.95` cosine。最终仍是一套
38-target、每target public rank16的LoRA，不是两套adapter。

## 4. Sealed generation evidence and its limit

80-row held panel（8 validation tasks × 10 videos）上的generation-only artifact：

`runs/outputs/pi05_reward_qv_pivot_rank14_plus2_transport_v1_e3857f7_20260811/analysis.json`

已经给出：

- 288个q/v pair-batches、16个action pairs；kept B bit-exact、zero residual B exact、finite，
  2880/2880 carrier A rows nonzero；
- q/v base error-to-zero mean约`.0007523`，每task最大不超过`.001302`，base cosine约`.9999997`；
- q/v rank2 tangent capture约`.9997088`；
- dynamic effective energy-weighted cosine约`.9975247`、error-to-zero约`.07218`；
- task-common cosine约`.99845`，video-centered cosine约`.95056`，centered per-video mean约`.93390`；
- action exact。

该artifact的`policy_action_forwards=0`、`rollouts=0`、`training_updates=0`。它只证明生成几何过门，
不能声称policy action或closed-loop改善。q/v base reconstruction error相对Reward tangent仍约
`1727x`，所以必须用真实行为裁决，不能再以几何替代。

## 5. Ordered behavior gates before any training

### Gate A: vertical native closure and throughput

在一张live空闲A40上，用同一32-request、同一总sampled-frame panel依次实测真实B8/16/32；从稳定、
不OOM且有可用显存余量的候选中选择端到端LoRAs/s最高点，不机械固定B8或最大batch。随后以所选batch完成：

- 真实Writer生成、原生72 BF16 + 4 FP32 cache、public adapter load均通过；
- 同一cycle1 condition比较old rank16、rank14 base、rank14+2三臂fixed-action before/after，必须证明q/v
  residual在实际LoRA matmul与policy accumulation后非零且方向一致，并覆盖四suite；
- no-video必须返回source identity；correct-video zero-Program只要求两个residual B slots和其incremental motion
  exact zero，不能把rank14 base误写成identity；
- generation不为了低位精度重复forward、扩dtype或缩batch；
- 记录LoRAs/s、peak allocated/reserved、host transfer和release。若吞吐下降，先优化batched compact
  pivot/solve/SVD；不得用更低吞吐换逐元素精度。

该门只允许窄smoke，不选择方法。

### Gate B: new macro0 strict correct400

先只评估q/v rank14 pivot-preserving base，Program为零、action保持原v6。必须使用与historical-v6
macro400完全相同的official paired validation 8×50 panel、state、policy RNG和one-video schedule。

这里的容量参照是同一paired schedule下旧full-rank zero-Program macro0的`134/400`，不是不同recipe的
历史峰值143。直接reject任一条件：

- correct `<130/400`；
- 相对旧full-rank macro0 `134/400` lost `>10`；
- breadth `<6`；
- 多suite出现广泛净退化。

该结果建立新compiler自己的真实base；不能用`.075%` reconstruction error推定identity。

### Gate C: current cycle1 Program load-only strict correct400

只有Gate B保留base后，才把已正式训练的Reward cycle1 Program通过同一compiler做load-only correct400；
不先训练、不改Reward scale/K/Nmc/RLS参数、不挑checkpoint。

- 若低于Gate B macro0，说明已学Reward方向即使原生可达也没有闭环价值，拒绝当前Reward方向/compiler组合。
- 只有`correct>=144`、breadth`>=6`、相对Gate B gained>lost且lost`<=6`，才算load-only通过；立即补
  同checkpoint完整controls，并可据此另写一次可导continuation design。
- `140--143`只是诊断性非通过：保留native transfer与任务换手证据，但不授权controls、任何其他扩展或
  新训练，也不能写成性能提升。
- 严格`>150`后仍必须完成correct/same/wrong/shuffled/reversed/no-video严格配对裁决，不能只报correct。

评测使用cost-balanced dynamic queue、long-first、persistent model/env。启动前实时比较`gpu01/gpu02`，
选择空闲卡更多且可有效提高吞吐的单个节点，并使用该节点当时所有真正空闲的A40；没有6卡上限，也不等待
凑卡或做dummy occupancy，不为跨节点碎片改写launcher。当前评测无NCCL；未来多卡训练仍须遵守
`NCCL_P2P_DISABLE=1`、NUMA physical/local rank映射和deferred-NCCL，且不得触碰有他人compute process的卡。

## 6. Training boundary after load-only evidence

在Gate B/C前不实现或启动新训练。若Gate C通过，下一设计必须解决fresh-zero可导性：

- forward使用本文原生compiler；
- backward使用明确的continuous analytic surrogate/STE，使native forward保持真实，而Program在
  `M=0`时仍有非零VJP；
- 不对全零tangent直接做SVD backward，因为重复零奇异值处梯度未定义；
- 不让`A_carrier`与`B_res`同时从零导致首步双线性零梯度；
- 不扩LoRA cache dtype、不降低B8或设备并行去追逐数值精度。

训练仍从frozen-v6起点、train24 task-complete宏步和one-shot信息墙开始。具体optimizer、resume schema、
profile与formal root必须在届时的新design authority中封存；本文不提前授权。

## 7. Historical non-equivalence and retirement trigger

本设计只有同时满足“condition-local carrier + physical zero-B slots + q/v-only rank reservation + action exact”
才是新结构：

- 固定global carrier会退化为已失败的SFT-Anchored Tangent Basis；
- task-expert carrier/route会退化为已失败的Expert-Manifold deployment；
- absolute full-rank refactorization会退化为已被本次native jitter直接否定的compiler；
- global scale、dither、gauge或local-CD均已由直接证据关闭。

若Gate B失败，退役pivot-preserving rank14 base；若Gate B过而Gate C低于macro0，则退役当前stable Reward
direction在rank-reserved compiler上的load-only组合。负结果不能扩大成“视频无用”或“Reward-Credit整体
无效”，但也不得通过扫rank、scale、dtype、seed或小panel重新解释。
