# On-Policy Success-Guarded Program Credit

状态：2026-08-11唯一active successor design authority；canonical实现与CPU/synthetic门已完成，尚未profile、
formal训练或评测。简称OSG-PC。
它从PICK-GC formal non-pass后最早失效的credit/occupancy接口出发，只改变task-local Program update的credit
约束；PICK-GC ordered goal-causal key、historical v6-fast frozen base、单一FP32 Program memory、full48
correct/negative solve、原生38-target rank16 compiler、B20 source-action proposal和one-shot部署图全部保持。

## 1. Decision and single variable

PICK-GC macro10相对immutable macro0得到`20 gained / 16 lost`。这说明blind B20 source-action cotangent包含
有用方向，但没有保护当前policy在真实occupancy上已经成功的support。OSG-PC不把这条可部署的proposal换成
另一个微小reward gradient，而是在每个train task的Program空间给它加一个无权重、可解析的可行域：

> 更新不得在一阶上增大当前正确视频LoRA已经成功执行过的on-policy action-prefix loss。

binary reward只选择哪些真实rollout是support；成功轨迹只形成“不破坏”约束，不成为新的正向模仿target。
失败轨迹不形成保护约束。新能力仍由同task跨episode B20 source-action proposal提供。

因此唯一主要变量是：

```text
PICK-GC:  blind offline Program proposal -> full48 write
OSG-PC:   blind offline Program proposal
          -> project into current successful on-policy occupancy feasible cone
          -> same full48 write
```

不改变rank、scale、seed、video数量、key、compiler、source objective、normalization、split或evaluation。

## 2. Why this interface is now the earliest failure

PICK-GC已经给出以下正式链条：

- raw/full48 key rank48、condition约`152.61`，correct/null与顺序结构可解；
- 10个macro的FP32 Program memory RMS=`3.5493e-6`且全部值非零；
- Program→LoRA→fixed action闭合；macro0→macro10 effective-BA相对L2中位`.002397`；
- strict却只有`138/400`、breadth6、retained/gained/lost=`118/20/16`、churn36。

所以不能再把问题归于condition collision、identity写出、LoRA能量不足或单纯compiler断路。RLS又证明固定
offline feature rows的保留不等于held rollout occupancy保留；Reward-Credit证明真实on-policy reward能形成
Program cotangent，但其q/v continuous factor motion约`1e-8`，低于native BF16局部ULP，cycle1仍为134且
`14/14`换手；uniform rank14为它腾physical slots又独立损伤base support。

OSG-PC继承这三项边界：

1. 保留PICK-GC已能跨native compiler的B20 proposal幅度，不重复sub-ULP Reward tangent；
2. 保护对象改为当前policy真实成功rollout的action occupancy，不重复offline-row RLS；
3. 直接写现有FP32 Program并复用完整rank16 compiler，不压缩、refactor或regenerate base。

历史Reward-Credit macro0同源K4 train24 panel为96 rollouts、60 successes，20/24 tasks至少成功一次；其中9个
tasks为4/4全成功、4个为0/4全失败。旧binary LOO把全成功与全失败都设为zero credit；OSG-PC恰好利用前者
作为已有support guard，同时让后者保持unconstrained improvement proposal。这个train-only统计只证明guard
有覆盖，不预测held性能。

## 3. Exact task-local graph

每个macro、每个train task仍只取一条correct ordered action-hidden teacher video，并与B20 action queries
跨episode错开：

```text
exact language + one correct ordered video
  -> frozen v6 evidence + PICK-GC goal/causal feature phi_i
  -> current FP32 Program residual at phi_i
  -> native complete rank16 LoRA

B20 cross-episode source actions
  -> current functional loss
  -> blind Program descent proposal d_i^0

same current LoRA + K4 official random-reset train rollouts
  -> binary success selects successful executed prefixes only
  -> per-success Program retention cotangents r_i,e

d_i^0, {r_i,e}
  -> exact Euclidean feasible-cone projection d_i^safe
  -> existing full48 correct d_i^safe / negative zero solve
  -> one shared FP32 Program memory update
```

K4是同一teacher video/同一LoRA在四个独立random-reset lanes上的训练occupancy，不是four-shot video。
deployment仍只运行一次Writer、读一条video、生成一套LoRA，rollout期间不再看video也不访问reward。

### 3.1 Blind improvement proposal

沿用PICK-GC exact B20 policy functional graph。设Program leaf为`H_i`，source functional loss为`L_i^src`：

```text
g_i^src = grad_H L_i^src
d_i^0   = -g_i^src
```

video/action query继续同task跨episode；B20先task内mean，24 tasks等权。step size仍为1，不增加global scale、
cap、momentum、Adam、weight decay或gradient clipping。

### 3.2 Successful occupancy cotangents

当前correct-video LoRA在四条official random-reset trajectories上执行。每条成功episode `e`保留自身真实
policy observations、归一化action chunks和实际执行的前1--5 actions；失败episode的replay立即释放，只保留
binary outcome与长度统计。

对每条成功episode，用历史已封存的Nmc4 keyed Beta(1.5,1) time/Gaussian noise panel、executed-prefix mask和
episode内chunk等权，计算：

```text
L_i,e^keep = mean_executed_chunk mean_m CFM_loss(current executed prefix)
r_i,e      = grad_H L_i,e^keep
```

这些actions来自policy自己的train24 rollout，不是teacher action，也不进入Writer输入或checkpoint。每条成功
episode独立形成一条half-space，避免把不同初始化平均后互相掩盖。它们只约束局部一阶变化，不要求Writer复刻
teacher轨迹，也不把失败动作当目标。

success-only collator仍保留四条lane各自的replan计数和原full-K4 row ordinal：先按历史task-key生成完整K4
flow panel，再只索引成功rows。失败observations/actions不保留、不forward也不求梯度；其长度只防止success集合
变化时让另一条成功episode的Monte Carlo样本错位。物理microbatch同样不能改变panel row identity。

为了显存与吞吐，correct video只encode一次。rollout使用detach后的current LoRA；retention VJP从已保存的
current Program input重新建立一次FactorHead decode leaf，不重复video encoder或source Writer forward。

### 3.3 Parameter-free feasible-cone projection

把Program tensors展平为`D=320*256`。对成功集合`S_i`定义：

```text
C_i = {d in R^D : <r_i,e, d> <= 0 for every e in S_i}
d_i^safe = argmin_{d in C_i} 0.5 * ||d - d_i^0||_2^2
```

`|S_i|<=4`。实现先把非零`r_i,e`做正尺度单位化；这不改变half-space。随后在FP64中枚举至多16个active
subsets，对小Gram使用确定性Moore-Penrose解，验证primal/dual/KKT并以active ordinal作唯一tie-break；大tensor
写回FP32。没有可调lambda、margin、trust radius或reward scale。

必须保持以下解析性质：

- `S_i`为空时`d_i^safe`逐元素等于`d_i^0`；
- raw proposal已满足全部约束时逐元素不变；
- permutation、重复或线性相关约束不改变解；
- 每条成功轨迹都有`<r_i,e,d_i^safe><=0`到FP64相对容差；
- 因`0 in C_i`，任一非零projection必有
  `<d_i^0,d_i^safe> >= ||d_i^safe||^2 > 0`，所以仍是source loss的一阶下降方向；
- 若唯一可行projection为0，该task安全地不写，而不是用scale强推。

### 3.4 Shared memory write

24个`d_i^safe`与24个exact-zero negative RHS仍按task ordinal固定排序，使用PICK-GC同一full48 condition
kernel、relative damping`.01`和FP32 Program add。negative schedule仍为8 reversed / 8 shuffled / 8 wrong，
且不做negative policy rollout/functional forward。无冲突task必须从guard到full48 application严格退化为原
blind update；不允许shared optimizer再把task方向混成common parameter update。

## 4. Why video knowledge and correct order remain necessary

OSG-PC不创建reward-only或language-only LoRA。所有proposal、成功guard和最终memory read都绑定同一
PICK-GC feature：

```text
phi(video) = normalize([
  terminal-quartile policy innovation - whole-video mean,
  centered causal-prefix policy innovation
])
```

zero-image subtraction使language本身没有condition value；static video给出zero feature。reverse把goal residual
从终态换到初态并翻转causal prefix，shuffle破坏terminal block或中间prefix连续性。训练只在correct ordered
feature处写有用`d_i^safe`，在真实reversed/shuffled/wrong feature处要求incremental zero。因此正确顺序不是
人为破坏negative LoRA的margin，而是访问reward-validated Program support所需的有向key。

同task不同video依靠PICK-GC已有same-task feature结构共享memory neighborhood；一条video仍不能保证鲁棒，最终
必须由same-task-other400验证。OSG-PC不增加第二条训练video来把credit与few-shot混成一个变量。

## 5. Coexistence argument and limits

OSG-PC同时处理两层换手：

1. task内：每条已成功initialization的executed-prefix gradient形成独立half-space，source proposal只能保留或
   改善其一阶loss；
2. task间：24个安全proposal在well-conditioned full48 key basis中同时写入同一个Program memory，negative
   conditions为零，不经过shared neural optimizer。

成功轨迹只是局部support proxy。train24 occupancy不等于held validation occupancy，CFM一阶不变也不保证离散
closed-loop success。因此该结构只比offline row retention更直接，不构成性能结论。paired400必须报告到底
保住了PICK-GC的哪些16 losses、是否保留20 gains，以及新出现的task/suite换手。

## 6. Rejected alternatives for this iteration

- **直接Reward Program cotangent**：已经以cycle1 strict134、`14/14`换手和q/v sub-ULP正式测过；更换key
  不能自动增加其native幅度。
- **finite Program ES/SPSA**：历史Program-Credit已能产生真实return gradient，但shared map失败；直接把探索
  perturbation写进memory仍缺少非任意update尺度，会同时改变credit与step geometry。
- **reward-signed offline proposal**：binary homogeneous tasks无法给符号，且只接受/翻转不能显式保护已有
  success support。
- **恢复RLS或rank14**：前者保护offline rows、后者损伤base；均没有针对本次最早接口的新证据。
- **few-shot、expert manifold或mixed rank**：可能仍有价值，但会同时改变representation、deployment或topology，
  不能与本单变量实验并行。

## 7. Implementation ownership and lifecycle

实现阶段必须原位替换唯一v6 residual training family：

- `expert_manifold/v6_prior_training.py`继续拥有task-complete graph、gather、full48 solve与checkpoint；
- `expert_manifold/v6_reward_credit.py`改为唯一success-retention VJP与cone projection owner，旧LOO Reward runtime
  不保留可执行分支；
- `reward/rollout.py`保留唯一K4 persistent-lane collector，新增success-only replay collator并在task结束释放
  failure replay；
- `v6_prior_contract.py`、checkpoint、inference/evaluator和唯一active config整体升fresh-incompatible schema；
- PICK-GC sealed config只作results provenance，不得继续被formal launcher接受；历史runtime由Git和formal
  artifacts保存，不保留strategy flag、fallback或第二CLI。

不新增第二trainer、第二Writer、第二full48 solver或大兼容层。若现有generic reward helpers没有第二用途，替换
时删除旧API/tests；results-only schema解析可保留。

## 8. Falsification gates

### 8.1 CPU and synthetic projection gate

实现后先验证：

1. 无success、raw-feasible、单冲突、多冲突、rank-deficient、duplicate/permuted constraints的解析解与KKT；
2. nonzero safe direction保持严格source descent，0 projection按合同停写；
3. success-only replay保留每episode chunk/valid-prefix/seed，failure replay不进入gradient；
4. teacher action/proprio/pose、validation/test action/reward reads均0；
5. correct graph只encode video一次，rollout detach与retention re-decode不改变Program/LoRA identity；
6. full48无冲突时逐字段退化为PICK-GC raw solve，negative RHS严格0；
7. checkpoint包含Program、K4 rollout cursor、teacher/action sampler、每rank RNG与exact world-size topology；
8. formal-result-sealed PICK-GC checkpoint/config不能误resume为OSG-PC fresh。

实现裁决：上述owner已原位替换，没有第二trainer、Writer、full48 solver、strategy flag或旧LOO可执行API。
success-only replay、per-success Nmc4 keyed CFM cotangent、0--4 guard解析KKT、raw-feasible/no-success exact fallback、
Program re-decode identity、full48 negative zero及fresh-incompatible checkpoint/config均有直接测试。加载
`.env.local`后的fresh完整CPU回归为`340 passed in 85.24s`；compileall与`git diff --check`通过。该结果只授予
下一节discarded live gate，不说明真实rollout guard覆盖、shared full48应用后仍可行或closed-loop有效。

### 8.2 One discarded full24 live gate

从historical v6-fast macro400、zero Program fresh运行一个完整train24 macro：每task一条correct video、B20 source
queries、K4 random-reset rollouts、success guards、full48 write，不保留checkpoint。live topology由launch前双节点
检查决定，task-complete仍须world size整除24。

2026-08-11 22:23+08:00的launch-preflight中，`gpu02`物理`0--5`均空闲且无active ECC/repair故障，故本次
profile封存为单节点world6/local4；物理`6--7`属于他人，不触碰。GPU1只有历史已纠正的DRAM/remap记录，当前
pending/failure均为no，发射前后仍须复核。matched world6/local4基线为`507.30541240703315s`，吞吐门仍是
`<=1.25x`。这只改变执行拓扑，不改变OSG-PC科学合同；实际launch前必须再次同时检查两节点、进程和quota。

必须同时满足：

- 24 videos、480 source queries、96/96 rollouts、四suite完整；至少18/24 tasks有success guard、至少6个
  4/4 all-success tasks、每suite至少2个guarded tasks；
- 每条成功episode retention cotangent finite/nonzero，失败replay gradient count0；
- 至少一个task的blind proposal违反guard且projection发生改变，否则方法退化为PICK-GC；
- no-success与raw-feasible tasks逐元素退化为blind proposal；全部safe directions满足KKT，所有非零方向保持
  source descent；
- full48 feature rank48、condition`<=200`、negative/correct motion`<=.15`、correct retained至少21/24、
  三类negative各null至少6且总计至少18；
- Program application、LoRA A/B、四suite source fixed-action response均非零，0 OOM/nonfinite/watchdog/
  forbidden reads/negative policy forward；
- report raw-vs-safe proposal norm/cosine、active constraints、all-success/all-failure、per-task source descent与
  success-loss first-order change，但不以漂亮数值替代closed-loop；
- 使用稳定且有显存余量的最大安全B20/replay batch，记录96-rollout wall、credit wall和峰值；不得为低位一致
  固定batch1、重复video forward或扩dtype。

任一hard gate失败即拒绝当前OSG-PC，不扫constraint margin、reward scale、K、Nmc、projection tolerance、
source step或key。

### 8.3 Formal and strict paired400

profile通过并由clean pushed seal封存后，从zero Program fresh训练`0→5`；每macro都重新采当前K4 occupancy，
不得复用stale success replay。checkpoint保存5与10，但首次只创建macro5。随后立即用与macro0 old134相同的
validation 8×50 without-replacement schedule做strict paired correct400。

macro5只有同时满足以下条件才允许exact-resume`5→10`：

- correct`>=140`；
- breadth`>=6`；
- 相对macro0 lost`<=8`且gained>lost；
- 至少两个suite净不降，aggregate不能由单task净增独占。

首次correct`>=144`立即补同checkpoint same/wrong/shuffled/reversed/no-video。macro10仍以严格`>150`为成功门；
若只到144--150则保留为高价值non-pass，不继续扫训练长度。最终goal checkpoint必须correct严格高于所有negative
controls、same-task-other/correct`>=.9`，且single checkpoint保持breadth与低换手。

## 9. What would falsify the hypothesis

- profile中blind proposal很少与success guard冲突：说明本次16 losses不是可由该on-policy tangent看到的
  support冲突；
- cone projection在train24满足约束，却让source descent或native action response普遍归零：success support与
  improvement proposal在当前Program coordinates不可共存；
- train success-prefix loss受保护但strict仍大量lost：train24 successful occupancy不能外推到held support，
  或CFM executed-prefix不是有效success tangent；
- retention改善但absolute无增长：guard解决了换手却没有改善proposal方向，下一接口才可能是reward-guided
  improvement而非support protection；
- correct与wrong/shuffle/reverse/no-video同步改善：更新主要依赖base/language或static shortcut，不能宣称视频
  教学有效。

负结果只淘汰“PICK-GC B20 proposal + per-success on-policy feasible-cone guard”这一组合，不否定所有reward
constraint、few-shot、task manifold或condition-local memory。
