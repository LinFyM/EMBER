# Success-Key Nullspace Consolidation

状态：2026-08-11 design authority已建立，尚未实现、profile、训练或评测。简称SKNC。
它从PICK-GC strict`138/400`与OSG-PC工程non-pass后最早仍开放的接口出发：保留PICK-GC已经接通的
ordered goal-causal key、historical v6-fast frozen base、单一FP32 Program memory、B20 blind source-action
objective、full48 correct/negative panel、原生38-target rank16 compiler和one-shot部署；只把“如何在共享Program
写入中保护真实成功support”改成condition-key空间的硬零运动约束。

## 1. Decision and single variable

PICK-GC macro10相对immutable macro0有`20 gained / 16 lost`。blind B20 proposal并非完全无用，但同一个共享
Program更新会覆盖已经成功的条件。OSG-PC试图从每条成功轨迹回放动作并求executed-prefix VJP；唯一world6
profile在full48前触发600s NCCL watchdog，wall lower bound至少`1.912x>1.25x`，所以该full-replay execution
graph已被工程门否决，且没有产生shared guard transfer证据。

SKNC只使用同一K4 rollout panel的binary outcome。若某个correct-video condition在四个独立random-reset lanes
上`4/4`成功，它成为一个on-policy success key。共享Program增量必须满足：

```text
phi_success @ DeltaMemory = 0
```

不是把该task的proposal乘以0，也不是训练confidence或global scalar gate。24-task B20 objective仍完整进入同一个
解析solve；实现把整个memory update参数化到全部success keys张成空间的正交补中，因而约束是在task-local
proposal汇合之后、完整shared write处成立。

唯一主要变量为：

```text
PICK-GC:
  full48 blind objective -> unconstrained Program write

SKNC:
  same full48 blind objective
  subject to current/persisted all-success condition keys having zero Program motion
  -> one constrained shared Program write
```

不改变key、rank、compiler、source loss、B20、video数、split、normalization、seed、LoRA scale或部署接口。

## 2. Why this is the earliest remaining interface

PICK-GC已经正式证明：

- ordered goal/causal evidence能区分same-task、reverse与shuffle，zero-image/static key为0；
- full48 feature rank48、condition约`152.61`，correct与三类negative可同时写/null；
- FP32 Program连续积累，Program→完整LoRA→fixed action闭合；
- effective BA不是identity：macro0→10相对L2中位`.002397`；
- strict结果却只有`138/400`、breadth6、retained/gained/lost=`118/20/16`、churn36。

因此representation、key conditioning、Program amplitude和native compiler都不是当前最早断点。历史RLS保护
offline feature rows仍未保护held on-policy occupancy；OSG-PC更接近occupancy，但其每成功episode动作回放/VJP
在机制结果前先违反吞吐合同。SKNC保留“reward只认证已有support”这一未被否定的假设，同时把保护对象从某些
状态上的action-loss一阶切面提升为整个video-conditioned LoRA本身：只要该condition的Program read不变，完整
LoRA和它在所有状态上的policy函数都不因本次memory write而改变。

这仍不是held成功保证。train24 success key可能不能覆盖validation key neighborhood，单条4/4也可能只是局部
幸运证据；strict paired400必须裁决这种保护能否外推。

## 3. Exact training graph

每个task-complete macro仍对24个train tasks各取一条correct ordered action-hidden video和B20条同task跨episode
action queries：

```text
exact language + one ordered video
  -> frozen v6 evidence + PICK-GC feature phi_i
  -> current Program read
  -> one complete native rank16 LoRA

B20 cross-episode source actions
  -> unchanged blind Program cotangent g_i

same current LoRA + K4 official random-reset rollouts
  -> retain only success bit, lane/seed and episode length
  -> 4/4 selects current success key phi_i

stored first success keys + current 4/4 keys
  -> one hard constraint span A

24 correct cotangents + 24 negative-zero conditions
  -> same full48 objective, solved in null(A)
  -> one shared FP32 Program memory update
```

K4是同一one-shot video/LoRA在四个初始化上的训练occupancy，不是four-shot video。rollout结束不保留observation、
action chunk、executed prefix、flow noise或reward gradient；失败与成功episode都不做replay forward/VJP。deployment
仍只读exact language和一条video，rollout前一次生成LoRA，随后不访问video、reward、anchor bank或task ID。

### 3.1 Binary certificate

只有`4/4`成功才认证当前key。`1/4`到`3/4`说明该condition仍不稳定，不能冻结；`0/4`没有support可保护。
这个阈值不是事后调参：historical同源K4 panel为96 rollouts、60 successes、9个4/4 tasks、4个0/4 tasks，且
四个suite均有4/4覆盖。它提供足够但不饱和的首个mechanism panel，并比“任一lane成功”更限制偶然初始化。

reward的唯一作用是产生这一个离散训练约束集合。它不决定proposal正负、步长、rank、route、target ownership
或checkpoint选择；validation/test outcome不进入训练。

### 3.2 Deterministic persistent anchor bank

checkpoint维护至多24条training-only anchors，每个封存train task最多一条：

1. macro开始时，先把已有bank keys和本macro全部current 4/4 keys组成约束集合；
2. 完成本次受约束写入后，若某task尚无anchor且本macro为4/4，则保存它的当前key；
3. 每task永远保留第一次按既定sampler/cursor出现的4/4 key，不替换、不打分、不按后续结果挑选；
4. current 4/4 key即使该task已有anchor，仍作为本macro temporary constraint，但不新增bank条目；
5. 0--3/4 task不新增或删除anchor。

train task ordinal只用于保证“一task一槽”和exact resume；它不进入Writer feature、Program read、LoRA route或
deployment。bank不保存第二套LoRA、expert、action或trajectory。首次anchor在写入前已进入约束集合，因此保存时
就是被本次write保护的值。

选择first-only而不是累计所有成功videos，是为了把状态固定在有界的24×256，防止reward越多就机械压缩可用
feature空间。same-task其它videos靠PICK-GC已测得的key neighborhood泛化；这项假设必须由same-task-other arm
验证，不能由bank大小替代。

### 3.3 Equality-constrained full48 solve

设Program memory为`M in R^(F x D)`，`F=256`、`D=320*256`。24个correct keys和24个negative keys按旧合同组成
`X in R^(48 x F)`；correct Program cotangents与negative zeros组成`Y in R^(48 x D)`。把persisted和current
4/4 keys去数值相关后组成`A in R^(m x F)`。

用FP64 small-matrix Moore-Penrose定义anchor row-span的正交补：

```text
Pi_A   = I - A^T (A A^T)^dagger A
X_perp = X Pi_A

DeltaM = -X_perp^T
          (X_perp X_perp^T + lambda I)^-1 Y
```

`lambda`仍是PICK-GC的`.01 × projected Gram mean diagonal`；若projected objective整体为0，则`DeltaM=0`而不
进入奇异solve。数值rank使用FP64 machine-epsilon×matrix-size的标准阈值，不引入可扫rank cutoff。

因为`A X_perp^T=0`，最终shared update直接满足`A DeltaM=0`。实现只对small rows做低秩投影，不物化或扫描
`256×256` dense projector；FP32大tensor写入前再做一次低秩roundoff correction，并报告实际
`A @ DeltaM`。无anchor时必须逐字段、逐元素退化为原PICK-GC full48 kernel。

这里all-success correct cotangent仍在`Y`中，objective没有task mask或乘零；其feature在可行子空间中自然投影
为0。若blind目标与success equality冲突，硬约束优先；其它tasks仍可通过与anchor span正交的feature成分共同
写入。negative RHS继续严格为0，且不做negative policy forward。

### 3.4 What is actually preserved

对于任一anchor key `a`：

```text
a (M + DeltaM) = a M                 # up to declared FP64/FP32 closure tolerance
```

PICK-GC的frozen v6 slots、FactorHeads与native compiler在两侧完全相同，所以该key下的Program residual、38-target
rank16 A/B、effective BA和policy函数都不应因本次memory update改变。该保护跨policy observation成立，不像RLS
只固定少量offline action rows，也不像OSG-PC只约束成功prefix loss的一阶符号。

有限精度下不预先宣称bit-exact。CPU/live gate必须同时测Program read closure、LoRA/effective-BA closure和同噪声
fixed-action closure；任一层明显放大roundoff即说明compiler接口不能把key equality变成有效保护。

## 4. Why video knowledge and correct order remain necessary

SKNC不增加reward-only Writer或language route。proposal写入、success认证、persistent anchor和部署read全部绑定
同一个PICK-GC key：

```text
phi(video) = normalize([
  terminal-quartile policy innovation - whole-video mean,
  centered causal-prefix policy innovation
])
```

video/action query继续同task跨episode错开，阻断逐帧动作复制。goal residual提取跨初始化仍成立的目标关系变化，
causal prefix保留阶段的有向连续性。reverse把终态残差换成初态并近似翻转causal项；shuffle破坏terminal block或
prefix连续性。static video两块均为0；zero-image subtraction后language本身没有condition value。因此：

- correct顺序决定哪个condition能获得B20正向write以及哪个已成功condition被保护；
- wrong/shuffled/reversed只作为full48 zero-RHS keys，不通过人为破坏LoRA制造margin；
- language/static输入不能访问非零Program residual；
- 任何高分若correct不优于controls，仍不能归因于video教学。

SKNC本身不改善PICK-GC representation。它只测试“已验证的有向key能否作为共享policy support的稳定地址”。

## 5. Policy-effective LoRA and coexistence argument

SKNC保留历史143架构与PICK-GC的有效子机制：

- historical v6-fast完整frozen base，不compression、SVD、refactor、regeneration或rank reservation；
- condition-local FP32 Program提供高于native BF16 factor局部ULP的连续写入；
- 原生38-target rank16 compiler、target ownership和跨层结构全部不变；
- B20 cross-episode source action仍提供已经实测能产生20 gains的policy-effective proposal；
- ordered correct key与三类negative full48 null结构不变。

新增的共存机制发生在最晚且最直接的shared-memory接口：每个macro先认证当前能闭环成功的condition，再让24-task
共同update只能使用不改变这些condition的feature directions。第一条成功anchor持续跨macro存在，使后续task/video
不能通过同一Program子空间覆盖它；本macro temporary success keys则阻止“先成功、同一步写坏”。

这是有限容量的可证伪假设，而不是免费午餐：anchor span可能吃掉有用方向，same-task key neighborhood可能不足，
train24 support可能与held无关。必须报告projected rank/energy、protected与unprotected motion、bank增长、per-task
success churn，以及strict retained/gained/lost；不能把零anchor motion本身当成方法成功。

## 6. Why this is not OSG-PC, RLS or a scalar gate

- **不是OSG-PC**：不保存成功trajectory，不读policy actions作target，不做Nmc4 replay、CFM VJP、task-local
  cone projection或per-success loss tangent。K4环境wall仍随真实episode长度变化，但rollout后的credit成本不再
  随成功episode长度、replan数或VJP数增长。
- **不是RLS**：RLS保护固定offline action response rows；SKNC保护由真实K4 success认证的完整conditioned
  Program/LoRA函数，并在最终shared solve上成立。
- **不是scalar gate**：没有`alpha_i`、confidence、reward weight或`d_i <- 0*d_i`。所有24个cotangents都进入
  同一objective；可行变量本身被参数化为`DeltaM in null(A)`。若冲突任务仍有正交feature分量，它继续更新。
- **不是task bank deployment**：anchor只约束一个共享memory的训练优化；held inference不查询bank、task ID或
  nearest route，也不携带第二套LoRA。

不在本轮同时加入few-shot、expert targets、rank/topology变化、reward improvement gradient、scale sweep或新的
video encoder，避免把support preservation与其它假设混杂。

## 7. Implementation ownership and lifecycle

实现必须原位替换OSG-PC唯一active training family，不保留并行runtime：

- `writer/condition_update.py`把现有full48 kernel泛化为唯一equality-constrained owner；无anchors exact退化为旧解；
- `expert_manifold/v6_prior_training.py`继续拥有task-complete gather、B20 objective、one shared Program write和报告；
- 删除OSG-PC per-success VJP/cone executable API，以一个小的success-key bank owner替换，不保留strategy flag；
- `reward/rollout.py`的当前active collector只产出outcome metadata，成功/失败均不保留replay tensors；
- checkpoint fresh schema必须拥有Program、24-slot anchor features/present mask/train ordinals、rollout cursor、
  samplers、每rank RNG和exact world topology；deployment adapter忽略training-only bank；
- OSG-PC sealed config保持fail-closed provenance，新建一个fresh-incompatible SKNC config；历史由Git、design和
  failure artifacts保存。

不新增第二trainer、Writer、compiler、full48 solver、checkpoint fusion或兼容fallback。实现前后的architecture
gate必须检查owner、caller、config和tests，旧OSG executable path须退役。

## 8. Falsification gates

### 8.1 CPU and synthetic gate

实现后必须先通过：

1. zero-anchor逐元素退化为PICK-GC；单anchor、多anchor、duplicate/permuted、rank-deficient和all-constrained有
   解析/高精度reference；
2. `A DeltaM`在FP32 application后接近roundoff，current/persisted anchors均零motion；
3. unprotected correct rows至少一个nonzero，negative RHS严格0，且无anchor时原correct/negative closure不变；
4. synthetic 24 persisted + 24 current anchors的bank-saturated case finite，rank/condition/closure可报告；
5. current 4/4 key在本次write前受保护，first-only bank deterministic，duplicate task不增槽，不替换anchor；
6. checkpoint/resume逐字段恢复bank、Program、cursor/RNG/topology；fresh SKNC不能误载OSG/PICK checkpoint；
7. 96个rollout record只含outcome/seed/lane/length等metadata，success/failure replay tensor和reward gradient count均0；
8. teacher action/proprio/pose、validation/test action/reward reads均0，negative policy forwards为0；
9. protected key的Program→LoRA/effective BA→fixed action闭合，unprotected key保持非零response；
10. repository只有一个active trainer/solver/config family，compileall、targeted和完整CPU tests通过。

任何基础性质失败即修正实现合同；若数学机制本身必须靠reward weight、anchor replacement、soft margin、rank cutoff
或task-specific route才能工作，则拒绝SKNC而不是sweep。

### 8.2 One discarded full24 live gate

从historical v6-fast macro400、zero Program、empty anchor bank fresh执行一个完整macro0：24 videos、480 B20 source
queries、K4=96 official random-reset rollouts、outcome-only certification、constrained full48 write；不保存checkpoint。
实际launch前重新同时检查gpu01/gpu02、进程、健康、quota与fresh root，选择单节点所有有益空闲A40；train24
world size必须整除24，NCCL/NUMA/deferred-init合同不变。

hard gates：

- 96/96 rollouts、四suite完整、总success与per-task K4可报告；至少6个4/4 tasks且每suite至少1个；
- outcome-only记录96条，success/failure trajectory replay tensors、replay policy/CFM forwards、reward gradients全为0；
  只允许预注册的shared-write后fixed-action closure forward；
- anchor bank新增数等于首次4/4 task数，task槽唯一；constraint count、numerical rank与suite覆盖一致；
- protected current Program motion相对unprotected correct motion`<=1e-5`，protected LoRA/effective-BA/fixed-action在声明
  tolerance内不变；至少一个unprotected correct condition和四suite各一个fixed-action response非零；
- projected objective rank等于数值reference，active nonzero spectrum的regularized condition`<=200`；unprotected
  correct projected/original feature-energy ratio中位至少`.20`，避免success span把剩余objective机械抹掉；
- negative/correct motion`<=.15`，未受保护correct中至少80%保留与自身cotangent正相关的descent motion，三类
  negative各至少6/8且总计至少18个null；
- full48 predicted/application closure、Program、LoRA A/B与action response finite；0 OOM/nonfinite/watchdog/
  forbidden reads/negative policy forwards；
- 从run-contract发布到完整report的wall不超过matched world6 K4 baseline
  `507.30541240703315s × 1.25`。该基线与OSG使用相同K4规模；SKNC不允许因成功轨迹数量增加额外VJP长尾。

`.20`只约束“还有可学习子空间”，不是性能代理；若这一固定首版被success span压平，不能降低门、减少K、软化
constraint或只保护挑选tasks。任一hard gate失败即拒绝当前SKNC，不重跑同配置或做小超参sweep。

### 8.3 Formal and strict paired400

live gate通过并由clean pushed seal封存后，从zero Program/empty bank fresh训练`0→5`，每macro重采K4 outcomes；
anchors只由既定first-4/4规则增长。macro5立即按immutable macro0 old134同一8×50 without-replacement schedule
做single-checkpoint strict paired correct400。

只有同时满足以下条件才允许exact-resume`5→10`：

- correct`>=140`、breadth`>=6`；
- 相对macro0 lost`<=8`且gained>lost；
- 至少两个suite净不降，aggregate gain不由单task独占；
- persisted anchors的Program/LoRA closure持续过门，projected feature energy没有单调坍缩。

macro10以strict correct`>150`为成功门。若仅144--150且lost`<=8`、breadth与suite趋势仍过门，保留为高价值
non-pass但不靠继续训练长度或anchor阈值sweep补救；回到最早失效接口设计下一单变量方法。首次`>=144`可补
same/wrong/shuffled/reversed/no-video以提前检查视频因果；最终goal checkpoint必须strict`>150`，correct严格优于
各negative controls，same-task-other/correct`>=.9`，且仍是同一single checkpoint。

paired分析必须逐task对比v6-fast143、macro0 old134、compiler138、online128与PICK-GC138，报告retained/gained/
lost、churn、breadth、suite、哪些anchors持续成功或失效；checkpoint union、task-specific选择和多checkpoint
融合不计。

## 9. Fast falsifiers and interpretation

- **几乎没有4/4 coverage**：binary certificate太稀，不能承担共存机制；不把1/4临时改成anchor。
- **projected feature energy/rank坍缩**：PICK-GC key把成功与待学方向放在同一span，硬保护与改善不可共存。
- **anchor key数值不动但LoRA/action明显动**：Program equality没有跨compiler，最早失败仍在compiler/interface。
- **train anchors保住而strict lost仍高**：train24 key neighborhood不能外推held support，或PICK-GC representation
  仍过于video-specific。
- **lost显著降低但absolute不增长**：support protection有效，下一最早接口是blind B20 improvement direction；
  不因此把SKNC写成超过150的方法。
- **correct与wrong/shuffle/reverse/no-video同步提高**：language/base shortcut主导，视频因果不成立。

负结果只淘汰“PICK-GC + first all-success key nullspace + blind B20”这一组合，不否定所有binary success constraint、
few-shot、task-level manifold supervision或reward-guided continuous improvement。

## 10. Cost, storage and recovery expectation

与OSG-PC相比，SKNC删除全部successful replay/VJP；GPU主成本应回到B20加96条K4 rollout，small-matrix projector和
最多48×256 anchors可忽略。首版仍按matched world6 K4 baseline预留约8.5分钟，hard上限约10.6分钟；正式每macro
按live topology重新估算，不用这份旧空闲快照预约设备。

anchor bank最多约24×256 FP32加少量metadata，checkpoint增量远小于1MiB；formal root预计仍由Program/checkpoint
主体主导，首次0→5峰值新增应低于2GiB，但发射前必须重新查`strg01 /data1`独立quota与实际root。checkpoint拥有
完整bank和sampler/RNG/topology，可exact resume；discarded profile不保留checkpoint。任何GPU工作只从clean pushed
commit的detached frozen worktree启动，并保留run contract、metrics/raw rows、completion或明确failure artifact。
