# Policy-Innovation Causal Consensus Key Writer

状态：2026-08-11 retained non-pass authority。PICK的raw-frame门通过，但discarded full48 profile的
regularized Gram condition=`483.61515`超过预注册上限`200`；其余14项机制、动作与吞吐门通过。
因此PICK未获formal训练资格，当前唯一后继是
`docs/action_forecast_writer_policy_innovation_goal_effect_key_design.md`中的PICK-GE。不得恢复本文旧“下一步”。

## 1. 决策摘要

下一候选只改变 frozen-v6 Program residual 的 **condition evidence owner**：

```text
旧 Balanced-v2 key:
  frozen-v6 Writer 的 train24-adapted frame evidence
  -> fixed balanced static/causal key

新 PICK key:
  frozen source policy 的 zero-image-subtracted high-level video innovation
  -> fixed balanced static/causal key
```

其余主要科学合同不变：

- historical v6-fast macro400的600个Writer tensors全部冻结；
- base Core、Procedure、slot compiler、八个FactorHeads和原生38-target rank16 LoRA不压缩、不重分解；
- 单一FP32 Program residual memory仍在frozen-v6 fused slots之后、FactorHeads之前写入；
- development仍是24 tasks、one-shot、B20、task-complete等权、同task跨episodeaction queries；
- 第一轮仍使用Balanced-v2的pointwise source-action Program cotangent和full48 correct/negative solve；
- negative只要求“本次incremental Program motion为零”，不把negative LoRA人为推坏；
- 部署仍只读exact language与恰好一条action-hidden teacher video并生成一套完整LoRA。

本方法简称 **PICK**：Policy-Innovation Causal Consensus Key。

## 2. 为什么这是当前最早失效接口

### 2.1 保留的正证据

1. v6-fast给出历史最好`143/400`，证明它的高增益base、Core/Procedure和compiler能产生policy-effective LoRA。
2. Balanced DC-Causal v2在不改v6 base的前提下得到macro0/10/25=`134/140/139`，13/13机制门、Program→BA→
   fixed-action和部署吞吐都闭合；因此“frozen v6 + condition-local Program memory”没有被否定。
3. Condition-Kernel证明显式condition memory能避免shared neural condition map把异质credit再次压成common
   update；它的失败是fresh decoder LoRA约比SFT小200倍，不是condition memory本身失效。
4. K4与Expert-Manifold的frozen source-policy innovation已证明视频、顺序和policy latent可被稳定读取；K4的
   失败是fresh M2P shared gradient retention约`1/24`，不是该frozen evidence必然无语义。

### 2.2 最新直接失败

Balanced-v2同task 50条正确视频的macro0→10修正方向近随机正交：raw LoRA delta一致性约
`.14154--.14218`，等于`1/sqrt(50)=.14142`；effective `Delta BA`与FP32 action-target pair cosine均约0。
macro0与10的success union为153，而single checkpoint只有140，说明模型已获得足以过151的分散能力，却没有
让同task不同video和跨macro修正在同一参数状态稳定共存。

RLS把offline feature-row漂移压低后仍为140且lost15；Reward-Credit替换为on-policy cotangent后仍为134且
14 gained/14 lost；rank14又分别证明compression和online regeneration会独立破坏support。因此下一步不应再
改solver、credit、rank、dtype或FactorHead，而应先让one-shot condition key表达跨video共同的高层有向程序。

## 3. Train-only design-selection evidence

现有sealed action-hidden cache：

`runs/outputs/pi05_expert_manifold_feature_cache_train24x50_r6_222d3ac_20260808`

覆盖24 train tasks×50 videos，每条为phase16×3072 BF16 frozen-source-policy innovation。本文建立前的只读
设计审计没有读取action、reward、validation/test或outcome，并按第4节候选公式直接统计：

| descriptor | same-task cosine mean/median | cross-task cosine mean/median |
| --- | ---: | ---: |
| static mean | `.99891/.99914` | `.83736/.85177` |
| centered causal prefix | `.91954/.92829` | `.15647/.14408` |
| complete balanced PICK key | `.95762/.96281` | `.49871/.49118` |

额外结构证据：

- causal correct-vs-reverse cosine mean/median=`-.97554/-.97742`；
- causal correct-vs-eight-shuffles cosine mean/median=`-.00529/-.00069`；
- complete key correct-vs-reverse cosine mean/median=`.01258/.01168`；
- 50个“每task同demo ordinal”correct24 panels全部rank24，`K+.01I` condition median/max=
  `89.87/108.56`；
- same-task leave-one-out centroid cosine中位：static`.99956`、causal`.96260`、complete key`.98083`。

这不是held结果，也不是正式新方法机制门。缓存用各视频自己的正确language，shuffle只在phase16 cache上近似
重排；wrong target-language grounding、真实raw-frame重排、online batch与新runtime都仍须重新验证。它只说明
“policy innovation比旧key更可能提供跨video共同的有向condition”有直接train-only依据，足以选择本单变量。

## 4. Exact PICK computation

### 4.1 Frozen source-policy innovation

对exact task language `T`先执行一次zero-image reference，再对每个stride-5 sampled frame独立执行同一
frozen source policy forward和固定`t=1` Gaussian Action-Expert suffix：

```text
h_t = concat(
    Mean_task_tokens(H_vl(frame_t, T) - H_vl(zero_image, T)),
    Mean_suffix(H_action(frame_t, T, fixed_suffix)
                - H_action(zero_image, T, fixed_suffix))
) in R^3072
```

这里不安装v6的Text/VL/Action Meta-LoRA，不读取teacher action。exact language只负责grounding；dynamic value
来自真实图像相对同language zero-image reference的innovation。no-video/zero-innovation必须精确为零。

每条可变长video在保持首尾和真实顺序的前提下线性resample到固定16个phase rows：

```text
x_0 ... x_15 in R^3072
```

这是现有`FrozenPi05VideoInnovationEncoder`与sealed cache的已验证语义，不新增第二种foundation descriptor。

### 4.2 Balanced static/causal key

保持Balanced-v2已通过的代数，只替换输入evidence：

```text
s   = Mean_p x_p
z_p = x_p - s
d   = Mean_p (Sum_{u<=p} z_u / sqrt(p+1))

u_s = ZeroL2(P_s s)
u_d = ZeroL2(P_d d)
phi = ZeroL2(concat(u_s, u_d)) in R^256
```

`P_s/P_d`为固定seed、无bias、逐row normalized的FP32 `[128,3072]` JL projection；不训练、不进optimizer、
不根据held结果选择。首版固定seed`20260810`，不做seed、feature width、phase数、static/dynamic权重或normalization
sweep。

性质：

- `s`保留对象、关系与任务内容，但单独不能满足correct→increment、reverse/shuffle→zero的full48合同；
- `d`只由真实顺序中的centered cumulative change产生；reverse近似翻转符号，shuffle破坏连续因果；
- 两个block等能，避免历史DC约50倍主导，也不逐频归一放大低能高频；
- zero innovation使两个block和完整key精确为零，不能形成language-only residual；
- 同一video只生成一个key、一个Program residual和一套LoRA，不做video或LoRA平均。

### 4.3 Counterfactual semantics

reversed/shuffled必须先改变真实sampled-frame content order，再执行phase resample和完整static/causal计算；不得
复用correct phase slots后只改标签。由于source encoder逐frame独立，训练热路径可在一次真实per-frame forward后
重排这些同一frame hidden再重算temporal descriptor，但必须用直接raw-frame两次forward的mechanism等价检查
封存；正式evaluator仍从实际重排后的frames完整生成LoRA。

wrong video继续使用目标task的exact language重新执行zero-image reference和video grounding。same-task-other是
正确分布，不作为negative。

## 5. Why the correct order is useful rather than negatives being damaged

PICK不训练一个“识别shuffle”的分类器，也不对negative adapter施加远离correct的margin。每个macro的正向
Program cotangent只来自correct video LoRA在同task跨episode B20 action queries上的functional descent方向。
full48 solve中的negative RHS严格为零，含义是本次正向增量不应泄漏到counterfactual key；negative仍保留同一个
frozen-v6 base，不被强行写坏。

所以若PICK有效，必须出现：

```text
correct ordered policy innovation
  -> stable same-task key neighborhood
  -> repeatedly accumulates a positive policy Program direction
  -> improves strict closed-loop

reverse/shuffle/wrong
  -> does not receive that ordered positive increment
  -> stays near the same frozen-v6 base
```

最终仍由同checkpoint correct相对same/wrong/shuffled/reversed/no-video的paired rollout证明；key cosine本身不能
支持视频因果claim。

## 6. Coexistence and base-support argument

PICK不要求不同task gradient在同一shared neural encoder中求均值。固定key与单一linear Program memory把每个
condition的credit写入其feature neighborhood；full24 solve显式处理当次24 correct与24 counterfactual rows。
与旧Balanced-v2相比，唯一预期变化是同task正确videos的keys从demo-specific neighborhood收缩到共同的
policy-semantic/causal neighborhood，从而让跨macro写入累积而不是为50条videos各写近正交方向。

base support保护为：

- historical v6的600 tensors逐tensor冻结；
- Program memory fresh全零时，完整LoRA必须与immutable macro0 native cache逐tensor相同；
- 不做rank14/15压缩、SVD、lstsq、gauge rotation、dtype扩展或factor refactorization；
- residual仍经过历史v6原生full-rank16 FactorHeads，避免fresh decoder低增益；
- no-video显式返回source-policy identity，不把v6 base误称identity。

因此最新rank14发现的compression损伤不应在本方法中重现；online regeneration仍以固定global request batching、
generator membership和cache contract控制，不能把worker数改变造成的BF16 batch差异归因于PICK。

## 7. Implementation ownership and lifecycle

保持一个canonical implementation：

1. 把现有通用`FrozenPi05VideoInnovationEncoder`从历史`expert_manifold`所有权移到`writer`下的单一policy-
   innovation owner；feature-cache consumer改用该owner，不保留import shim或复制实现。
2. `writer/condition_update.py`原位用PICK替换`FixedBalancedCausalConditionFeature`；旧Balanced-v2 key由Git
   `3a6f801`和formal artifacts保存，不保留并行runtime strategy。
3. `FrozenV6ConditionResidualWriter`拥有一个frozen policy-innovation encoder、一个fixed PICK projector和
   原有Program memory；base Writer与source policy均无trainable parameters。
4. `v6_prior_step.py`继续是correct/counterfactual graph owner；只改变feature构造，不复制training loop、
   evaluator或checkpoint family。
5. 新config/checkpoint schema必须fresh-incompatible；旧RLS/Reward/rank14 configs继续results-only fail closed，
   不得误载为PICK。

若PICK终局non-pass，删除其active config、runtime分支和专用tests；通用policy-innovation encoder可在仍有
feature-cache consumer时保留。历史由本design、Git和formal artifacts保存。

## 8. Evidence gates

### 8.1 CPU and cache gate

必须验证：

- zero-image subtraction、phase endpoints、shape/dtype/finite和zero key；
- reverse/shuffle重排真实content后重算；
- fixed seed/projector不进checkpoint和optimizer；
- step0 Program memory全零、source freeze、38-target rank16完整；
- fresh/resume cursor、Program memory、每rank RNG和world-size fail-close；
- sealed train24×50 cache重算第3节统计，无validation/test/outcome reads。

cache机制最低门：causal same-task cosine median`>=.85`、cross-task median`<=.30`、correct-reverse median
`<=-.85`、correct-shuffle absolute mean`<=.10`；complete-key same-task median`>=.90`且50个correct24 panels
全部rank24、`K+.01I` condition max`<=150`。任一不通过则不改阈值/seed/phase数，直接拒绝当前key。

### 8.2 Live mechanism and throughput gate

在一张live空闲A40上先用raw frames验证cache/online方向、zero-video、target-language wrong、真实reverse/shuffle、
reorder-hidden与raw-frame重跑等价。随后用完整train24 full48 discarded mechanism macro验证：

- feature rank48、condition finite且不高于200；
- task-local correct retention至少21/24，三类negative各至少6/8 null；
- aggregate negative/correct Program motion`<=.15`；
- predicted/observed application、A/B、四suite fixed-action和source freeze闭合；
- 0 OOM/nonfinite/forbidden reads/negative policy action forward。

generation profile使用相同32-request、真实longest-video panel测B8/16/32，选择stable、无OOM、LoRAs/s最高且能
覆盖B20 formal的候选。不得因普通BF16低位差异固定batch1、重复forward或扩dtype。若额外frozen innovation
forward使完整macro wall超过sealed Balanced-v2的`1.75x`，先消除重复frame encoding、host sync和小tensor传输；
仍无法满足则以部署/训练效率non-pass停止，不缩科学batch。

### 8.3 Real behavior gate

mechanism与deployment seal通过后只允许：

1. 复用immutable exact macro0=`134/400`作为paired baseline，同时做一次PICK zero-memory vertical identity；
2. 从fresh zero Program memory做blind Balanced update `0→10`；
3. 立即做同一8×50 schedule的strict correct400，不先做80-row screen。

macro10继续门：

- correct`>=144/400`；
- breadth`>=6`；
- 相对macro0 lost`<=8`且gained>lost。

任一失败即退役当前PICK+blind-credit组合，不转RLS、Reward、rank reservation，不扫seed、scale、lambda、phase、
feature width或dtype。三门通过但correct为144--150时，只允许一次exact-resume`10→25`并再次strict400；是否继续
更远必须由当时single-checkpoint absolute、retention和churn重新写authority，不能因“还能训练”自动续。

首次correct`>150`或达到`>=144`且三门全过时，立即补同checkpoint严格配对
correct/same/wrong/shuffled/reversed/no-video。最终成功仍要求correct沿有用方向实质优于四个negative和
no-video、same-task-other保持鲁棒；aggregate高但lost/churn或视频因果失败仍是科学non-pass。

## 9. Fast falsifiers

以下任一证据直接否定当前假设：

1. sealed cache统计不能由canonical实现重现，或online raw-frame key失去same-task/causal结构；
2. full48 condition病态、negative leakage重现v1，或Program不能到达BA/action；
3. zero Program改变immutable v6 base，说明新evidence path污染了base support；
4. 同task不同video的Program/effective-BA correction仍近随机正交；
5. macro10未过144/breadth/retention门，说明condition consensus不是当前closed-loop首因或blind AS credit仍主导；
6. correct提高但wrong/shuffle/reverse/no-video同样提高，说明收益仍是static/base shortcut；
7. 吞吐必须靠batch1、重复forward、扩dtype或牺牲B20才能运行。

## 10. Claim boundary

PICK若通过，只能说明“frozen policy-aware causal condition evidence + frozen-v6 high-gain residual memory”在当前
one-shot benchmark上改善了共同积累。它不证明所有task experts、few-shot、Reward或parameter manifold路线
正确，也不把train24 task grouping带到held部署。若失败，只淘汰当前frozen policy-innovation key与blind
Balanced credit的组合；不外推为视频、高层policy feature或所有cross-video consensus方法无效。
