# V6-LPCP Native Probe-Value Commitment

状态：2026-08-15 active design authority；尚未实现、profile、训练或评测。CCT已终局，不得resume其cycle1或把
本设计写入旧checkpoint。本轮从sealed LPCP macro25 fresh建立不兼容commitment，只改变**factor commitment读取
哪一个video-dynamic Value**。

## 1. 决策与最早失效接口

CCT在train-seen task4的纯CCT four-view effective-BA cosine/energy为`.575776/.681821`，但held validation
first4约为`0/.25`。exact evaluator worker已确认65,536个semantic-query元素逐元素完整加载。train→held的
transported coefficient与pre-W2 hidden residual只缩小`1.63x/1.70x`，pure-CCT BA L2却缩小`249.92x`。

所以当前最早失效接口不是视频读取、时序、K-set、checkpoint loader或reward credit，而是：

```text
ordered-video evidence
    -> LPCP-vs-AS139 tiny readout difference
    -> pre-W2 residual
    -> native BF16 factor compiler
    -> held effective BA nearly disappears
```

CCT把`Procedure_LPCP - Procedure_AS139`当作factor memory。这个量本质上是一个已经经过zero-init
`query_delta`、Procedure reader与K-set之后的二阶小差分；它在train task上偶然跨过大量native factor边界，held
条件只小约1.7倍却几乎完全退回LPCP。

本轮选择**Native Probe-Value Commitment（NPVC）**：保留CCT的language/policy-aligned transport与全部强底座，
但让factor commitment直接读取LPCP已经证明有序、layer/rank对齐且constant-zero的native Action-probe动态状态，
不再把微小LPCP-AS139 readout差分当作唯一Value。

## 2. 保留项与非目标

以下逐项冻结：

- exact language + dynamic `K=1..4` same-task ordered action-hidden videos；canonical formal仍用K4；
- stride5、真实frame ordinal、每video内部有序编码、跨video permutation-invariant集合语义；
- AS139强底座、LPCP同一次joint forward的18层Action-probe carrier与`query_delta`；
- 320个policy/rank slots、八factor families、38 targets、完整public rank16 LoRA与native FactorHeads；
- source policy、normalization、train24/validation8 split、four-view selected-success objective、optimizer、rollout数、
  dtype与信息墙；
- step0 exact LPCP，language不能单独产生NPVC residual，source policy没有trainable parameters。

本轮不加入literal memory token、rank8/rank18、额外factor lane、output normalization、global scale、coherence loss、
expert、contrastive、negative margin、B20 functional loss、更多reward rollouts或第二数据源。它不是一次“放大CCT”
的参数小扫，也不改变LoRA容量。

## 3. 已有native probe Value

对第`k`条视频，LPCP在同一次真实image + exact language + 50 Action-probe joint forward中旁读18层hidden，并由
`LayerwiseProbeProcedureConditioner`得到：

```text
H_k ∈ R^(320 × 256)
```

这里320对应`18 Action-Expert layers × rank16 + action-in rank16 + action-out rank16`。`H_k`不是新memory token；
它是已经存在的policy-layer/rank-aligned动态状态：

1. 每帧先读取真实prefix条件下的native Action-probe hidden；
2. 同一video内先做相邻frame delta，首帧delta为0；
3. causal temporal controller使用真实frame ordinals；
4. 每个layer/rank slot只汇聚自身有序delta；
5. constant-frame video使所有probe delta与`H_k`精确为0。

LPCP已证明该carrier在reverse/static/one-forward门上工作，并可保持143底座。本设计不为了使用memory而替换这一
已通过的carrier；SHINE的可取原则是复用原生context中的逐层state做结构化参数生成，Doc-to-LoRA的可取原则是
共享layer/module/rank-aware readout，而不是必须具有某个token名字或照搬文本模型payload。

参考：

- SHINE: https://arxiv.org/abs/2602.06358 ，https://github.com/Yewei-Liu/SHINE
- Doc-to-LoRA: https://arxiv.org/abs/2602.15902 ，https://github.com/SakanaAI/doc-to-lora

## 4. 跨video probe-Value聚合

NPVC不新增另一套set网络。对同一condition，继续先按LPCP完整计算每条video的query-conditioned Procedure和
已经封存的`PolicyProcedureCommonValueFusion` attention：

```text
alpha_k,s >= 0
sum_k alpha_k,s = 1
```

其中`alpha`对video permutation等变、对最终集合归约不变，并且按policy slot分别读取。NPVC用**同一份attention**
聚合native probe Value：

```text
M_native[s] = sum_k alpha_k,s * H_k[s]       # [320, 256]
```

这不是平均分别生成的LoRA，也不是另算一个best-video selector：每条video先独立形成完整有序`H_k`，set attention
来自同一condition的Procedure证据，最后只生成一套LoRA。K1时`alpha=1`。video permutation只重排求和项，不改变
结果；每条video内部reverse/shuffle会先改变probe delta、causal controller与Procedure attention。

## 5. language给方向，video给Value

后半段沿用CCT已通过的结构。对每个condition和slot：

```text
L  = text-only V6 Core的slot-aligned exact-language state
u0 = RMSNorm(L)
u1 = fixed_sign ⊙ u0
c0 = <M_native, u0> / sqrt(256)
c1 = <M_native, u1> / sqrt(256)

q = Wq RMSNorm(L)
p = softmax(q RMSNorm(K)^T / sqrt(256))       # p0...p3

h_f0 = GELU(W1_f u0)
h_f1 = GELU(W1_f u1)
R_f  = (p0-p1)c0 h_f0 + (p2-p3)c1 h_f1
```

`W1_f`与最终`W2_f`仍是冻结V6 FactorHead；`Wq`zero-init，故step0 `p0=p1=p2=p3=1/4`，`R_f=0`且输出逐tensor
exact LPCP。trainable仍只有semantic query、basis keys与两个RMSNorm，共`67,072`参数。

NPVC没有language-only LoRA branch：`H_k=0`时`M_native=0`，无论language是什么都严格没有新增residual。language
只规定同task共享的policy axes；video的有向probe Value决定沿这些轴写多少。与CCT相比，唯一变化是
`M = tiny(Procedure_LPCP-Procedure_AS139)`变为`M_native = set(native ordered probe deltas)`。

## 6. 为什么不是历史路线重跑

- **不是Dynamic-K Backbone-Memory rank8**：不追加8个literal tokens，不丢弃V6 Semantic Core，不换fresh rank8/
  fixed-A mapper；LPCP143、rank16 native compiler与完整base Program逐项保留。
- **不是把LPCP carrier再证明一次**：carrier的reverse/static证据已经成立。本轮只检验已读出的`H_k`能否作为
  direct Value越过held commitment，而不是只在Procedure Query里产生一个微小差分。
- **不是SFMC/GOSC/CCT scale sweep**：不乘常数、不归一化到目标norm、不增加axis。Value tensor来自更早且有明确
  policy语义的接口，参数量与transport公式不变。
- **不是B-only residual或rank reservation**：38-target rank16 A/B仍由同一V6 FactorHeads完整生成，没有压缩、
  append lane、第二套adapter或template refactorization。
- **不是language shortcut**：probe delta为0时新增写入严格为0；wrong/shuffled/reversed/no-video资格仍必须由
  closed-loop controls证明。

## 7. 实现与fresh合同

canonical Writer原位替换CCT factor-memory语义，不保留并行runtime class。checkpoint schema必须fresh-incompatible：
CCT `factor_commitment`即使tensor shape相同也不得加载，因为同一权重在不同Value定义下没有resume语义。

实现只需：

1. 在`compile_readouts`保留每个condition的现有Procedure-set attention；
2. 以该attention聚合`per_video_query_conditioners`形成`M_native`；
3. candidate Program仍是LPCP，reference仍是同cached condition关闭`query_delta`的exact AS139；
4. factor commitment读取`M_native`，其余decode、reward、evaluator与checkpoint owner不变。

必须先做定向CPU合同：K-set permutation、K1、constant probe zero、step0 exact LPCP、CCT checkpoint拒载、
factor family/slot ownership、reference/candidate语义与reward/evaluator一致。

## 8. formal前train→held快速否决

本轮最关键变化是：**held gate前置到formal之前**。一个train task smoke通过不再足够。

先在一个train task做一次真实selected-success update，再用同一post-update state只读视频地生成：

- train task4的4个disjoint correct K4 conditions；
- validation8每task 4个disjoint correct K4 conditions；不读取validation actions、reward或outcome；
- task4 natural/reversed/constant conditions。

必须同时满足：

1. step0逐tensor exact LPCP；base/LPCP/source参数0 gradient，semantic query首步finite/nonzero；
2. train task4 q/v/action native BA与fixed synthetic-action response均非零；
3. train task4 pure-NPVC four-view aggregate cosine至少`.15`、mean/sample energy至少`.40`；
4. held 8 tasks中至少6个的four-view cosine至少`.10`、energy至少`.35`，8-task aggregate cosine至少`.15`、
   energy至少`.40`；不得再次出现整体约`0/.25`；
5. held/train pure-NPVC effective-BA L2 ratio至少`.10`，且held mean relative-L2至少`1e-4`；这只要求离开CCT的
   `1/250`断裂，不以内部幅度选择最终方法；
6. natural→reversed的probe Value与NPVC LoRA material变化；constant新增BA norm不超过natural的`1e-3`；
7. longest-video cycle wall不慢于CCT`1.10x`，无OOM/nonfinite，按recorded wall判断rank负载。

任一held项失败即终局，不启动full24。门只证明结构没有在train→held接口立即失效，不提供absolute、稳定性或
视频因果closed-loop结论。

实现状态（2026-08-15）：canonical Writer已原位切换为fresh-incompatible NPVC schema。`compile_readouts`用
Procedure-set返回的同一attention对`per_video_query_conditioners`做slotwise K轴加权，所得
`shared_probe_value_slots`直接进入原factor commitment；LPCP/AS139 reference差分不再充当Value。constant-video
zero、natural非零、K-set permutation、attention逐项等式、step0 exact LPCP与cold-start拒载合同均已覆盖；
定向CPU=`43 passed`，完整CPU在canonical LIBERO assets环境=`398 passed`。尚无GPU机制或性能结果。

## 9. 训练、strict与稳定资格

机制门通过后，从sealed LPCP macro25 fresh进行与CCT完全matched的full24 cycle1：每task两组paired states，
candidate/reference唯一成功trajectory在4个disjoint correct K4 conditions上完整CFM，先task内四view等权，再
active-task等权。ties与both-success/fail为零credit。不加入额外loss或rollout。

cycle1后立即做single-checkpoint K4 strict paired correct400。只有同时满足才exact-resume cycle2：

- correct至少`140/400`、breadth至少7；
- 相对LPCP143 lost不超过15；
- 没有suite灾难性清空；
- held post-train four-view gate仍通过，且gained改写不能系统性小于lost。

owner认可的稳定约145资格保持：

- cycle1/cycle2都至少142，两点均值至少145，breadth都至少7；
- 相邻checkpoint churn不超过20、Jaccard至少`.85`；
- final相对LPCP lost不超过10、gained不少于lost，多task而非单suite净积累；
- 不用checkpoint union、per-task选点、LoRA平均或多checkpoint融合。

只有稳定资格通过，才对同一final checkpoint做strict paired correct/same-task-other/wrong/shuffled/reversed/
no-video。same-task-other不低于correct超过8分；correct相对每个negative/no-video至少净高10分，且paired
gained>lost覆盖至少3 suites。单点145或151都不能跳过相邻稳定性与视频因果资格。

## 10. 快速否决与负结果边界

- native probe Value在held仍sub-threshold：否定当前direct-probe + CCT transport，不继续scale/normalize；
- held写出material但four-view仍约`0/.25`：说明probe Value本身video-local，不加coherence loss救；下一接口才可
  考虑literal memory/set representation；
- held机制过门但strict下降或高lost：说明输出可达但方向不对held occupancy，不以LoRA健康度继续训练；
- cycle1尚可但cycle2高churn：说明direct Value没有解决shared credit共存，应改变训练/optimizer接口；
- stability过门但六臂无margin：说明方法主要沿用LPCP/language能力，不具有效视频教学资格。

任何失败只淘汰本轮“frozen LPCP native probe Value + existing Procedure attention + CCT transport + one-cycle
selected-success”组合，不否定V6/LPCP、literal memory token、rank8、few-shot、reward credit或生成LoRA。
