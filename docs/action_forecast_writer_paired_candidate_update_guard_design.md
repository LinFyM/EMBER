# Paired Candidate-Update Guard

状态：2026-08-12 active design authority，尚未实现、profile、训练或评测。PCUG只改变SKNC最早失败的shared-credit
接口：先形成原始shared blind candidate，再用严格配对的closed-loop base/candidate outcomes识别这次candidate
实际会伤害的tasks，最后在同一次memory write前把这些task的condition motion投影为零。它不做policy gradient、
trajectory replay、landmark imitation或reward-scale sweep。

## 1. Why this interface

当前连续证据是：

1. v6-fast证明historical Writer、task-complete B20 recipe和native rank16 compiler能达到`143/400`；
2. PICK/PICK-GC证明ordered video condition、Program、effective LoRA和action链路能工作，但blind shared update仍换手；
3. SKNC把all-success train key的最终Program motion压到零，strict仍`137`且lost13，说明“成功过一次”不是当前
   candidate是否伤害held support的充分证据；
4. OSG在task-local success guard到shared solve之前失效，说明guard必须作用于最终shared write；
5. SRTP试图取得完整reward Program gradient，但释放decoder graph后的唯一reprofile仍在logical B<=16 CFM
   backward三rank OOM，尚未产生shared half-space证据。

因此下一步不再近似一个昂贵的完整reward gradient。更直接的问题是：**当前full24 blind candidate在真实闭环中，
相对当前checkpoint究竟使哪些task变差？** 这个一维因果问题只需要严格配对的两臂rollout，不需要对policy反传。

## 2. Single causal variable

PCUG保留SKNC以下部分不变：

- historical v6-fast frozen Writer初始化、PICK-GC ordered condition key和negative zero-RHS full48；
- exact language + one action-hidden correct video、stride5、Q/M/G、Core/Program和38-target rank16 compiler；
- 每task一条video、B20同task跨episodeaction queries、task内mean后full24等权；
- FP32 Program memory、relative-damped blind solve、first stable-success key bank、optimizer和部署图。

唯一主要变化是把“当前4/4 success就保护”升级为“先对实际candidate做paired causal test，再保护candidate明确伤害的
task”。reward只决定最终equality guard集合，不产生连续梯度、不设可调scale，也不进入Writer输入。

## 3. Two-phase macro

### 3.1 Phase A: blind evidence and provisional candidate

每个train task只读一条correct action-hidden video，生成当前Program `H_i`、完整LoRA、correct/negative condition
features和B20 functional Program cotangent `g_i`。先汇合完整24 tasks，再只使用**此前已persist的stable-success
keys**按sealed SKNC arithmetic形成provisional shared candidate `D0`。

fresh macro0的bank为空，因此`D0`必须与相同inputs下的无anchor SKNC/PICK-GC blind update逐元素相同。后续macro
只让先前checkpoint中已封存的stable-success keys进入provisional solve；当前macro的reward尚未观察，不能提前
影响candidate。

### 3.2 Exact candidate Program

对task `i`计算candidate motion：

```text
delta_H_i = phi_i D0
```

candidate Program必须复现真正commit `D0`后的数值顺序：

```text
H_i_base = base_slots_i + cast(residual_i)
H_i_candidate = base_slots_i + cast(residual_i + delta_H_i)
```

训练图需保留detached `base_slots_i`和FP32 `residual_i`，不能用`H_i_base + delta_H_i`冒充上述次序。两臂都只
重解native compiler；不重读video、不重新计算condition、不改变rank或scale。

### 3.3 Two paired initializations, two arms

每task固定两个random-reset initializations。对每个初始化，base arm使用`H_i_base`，candidate arm使用
`H_i_candidate`；两臂严格共享：

- env seed与reset state；
- rollout cursor；
- policy-noise root和每次replan noise seed；
- dummy settling、horizon、action execution和inference steps。

base K2与candidate K2顺序执行并复用persistent env lanes；总计仍是每task四条rollouts、full24共96条，不增加
相对SKNC K4的episode数。两臂允许正常BF16/TF32低位差异，因为这种差异正是candidate deployment数值的一部分；
不得为了逐元素一致改batch1、扩dtype或重复arm。

对paired state `j`定义：

```text
loss_ij = 1[y_base=1 and y_candidate=0]
gain_ij = 1[y_base=0 and y_candidate=1]
```

task分类固定为：

- `harmful`：`sum_j loss_ij > sum_j gain_ij`；
- `beneficial`：`sum_j gain_ij > sum_j loss_ij`；
- `indifferent`：两者相等；
- `stable-success`：base与candidate四个outcomes全部成功。

不按aggregate success fraction、confidence、margin或连续reward调阈值。相同state上的candidate loss才是写入伤害
的因果证据；不同初始化之间的success差异不能互相抵消成伪方向。

## 4. Final shared guard

令`A`为persisted stable-success keys，`G`为当前macro的：

```text
G = current stable-success keys union current harmful keys
```

最终update是离`D0`最近、同时满足全部最终equality的Euclidean projection：

```text
min_D 0.5 ||D-D0||_F^2
s.t.  A D = 0
      G D = 0
```

`D0`已经满足`A D0=0`，所以实现只需对`[A;G]`做FP64小SVD/orthonormal basis，并在GPU FP32对21M-value
candidate做一次feature-axis projection。不得重新求一套full48 acquisition solution；否则新增guards会改变所有
未guard tasks的方向，重现OSG“guard后又被shared solve改写”的接口。

性质：

- 无current guards时，final `D`必须与`D0`逐元素相同；
- 每个harmful/stable task的最终`phi_i D`必须为零；
- beneficial/indifferent tasks只承受满足guards所需的最小全局改动；
- projection与task order、重复/相关guard rows不变；
- final update与`D0`内积必须为正，不能通过反转blind acquisition满足保护。

只有`stable-success` keys按first-success policy进入checkpoint bank；harmful guards只约束当前candidate，不持久化。
因为harm是update-specific，永久保存会把一次局部冲突误写成task终身冻结。每个macro仍重新probe完整24 tasks。

## 5. Why this can learn high-level task knowledge

PCUG不监督teacher轨迹。video仍只经ordered evidence/condition/Program决定当前LoRA及shared memory address；B20
actions与video跨episode错开。paired reward发生在当前policy对两个未见初始化的闭环执行上，只问同一个
video-conditioned shared update是否降低任务完成率。

路径、速度、抓取角度等单demo nuisance若不改变跨初始化成功，不会形成guard；真正破坏对象关系、目标状态或阶段
顺序的update会在相同初始化的base/candidate对中表现为loss。不同macro重采video与state，shared memory只有在
多个video-conditioned addresses上不产生可观察伤害时才能累积。

## 6. Correct order and anti-bypass

- Writer仍没有language-only LoRA value path；video是condition与Program的唯一dynamic value；
- exact language只作为video evidence query，不能单独生成task LoRA；
- correct/reversed/shuffled/wrong features继续以PICK full48 zero RHS参与blind `D0`，保持有向过程约束；
- paired probe使用由correct ordered video生成的base/candidate Programs，reward不读取teacher action/state；
- training task ordinal只拥有sampler、bank slot和evidence记录，不进入Writer或deployment；
- validation/test reward、actions和task ID从不参与训练。

若formal absolute过门但correct不优于negative/no-video，方法仍科学non-pass；paired guard不能替视频因果性背书。

## 7. Coexistence mechanism

SKNC只保护“这条train video曾4/4成功”的静态地址；PCUG保护“当前真实shared candidate会把成功变失败”的动态
干扰证据。guard位于final memory write，故不会被后续task汇合改变。closest projection又避免因保护少数task而
无理由重算或旋转其余tasks的blind direction。

这仍是acquisition-plus-protection方法：

- B20 full24 blind `D0`负责获得新能力和幅度；
- stable-success bank负责长期保留已稳定support；
- current harmful guards负责阻断本步可观测能力丢失；
- beneficial/indifferent tasks不被reward直接写入或放大。

## 8. Fast falsification

### 8.1 CPU and synthetic

1. base/candidate完全相同state、env/policy seeds，交换queue/worker order不改变pairing；
2. exact candidate Program按`base_slots + cast(residual + motion)`生成；
3. fresh no-bank provisional `D0`逐元素退化为sealed blind update；
4. no-guard final `D`逐元素等于`D0`；
5. single/multiple/correlated/duplicate guards的final motion为零，projection具permutation invariance；
6. final/D0 energy、inner product和cosine为正，persisted anchors保持zero；
7. harmful只在paired losses严格多于gains时成立，stable-success必须四臂全成功；
8. harmful不进checkpoint，stable-success first key、Program/cursor/sampler/RNG/topology exact resume完整；
9. reward path没有policy backward、landmark/action replay或teacher forbidden reads；
10. 一个trainer、一个shared solver、一个deployment path，SRTP runtime不可resume。

### 8.2 One discarded live macro

从historical v6-fast、zero Program、empty bank运行train24 macro0：24 videos、480 B20 queries、48 exact paired
states、96 total rollouts。按live同节点有效A40数运行，至多6、不等待凑卡；多卡继续deferred NCCL、P2P off和
NUMA physical/local mapping。

必须同时满足：

- 48/48 pairs的state/env/policy seeds exact，base/candidate各48 rollouts；
- candidate Program motion和native LoRA/action response在四suite非零；
- paired discordant states`>=4`、harmful tasks`>=2`且至少2 suites、candidate gains`>=1`，否则guard没有内容；
- final guard rows等于stable-success union harmful，final violations为0且projection实际改变`D0`；
- final/D0 energy ratio`>=.25`、inner product/cosine为正、projected feature rank`>=24`；
- persisted/current guard Program→LoRA→fixed-action closure过门，unprotected四suiteaction response非零；
- full48 negative/correct motion ratio`<=.15`，0 forbidden read、OOM、nonfinite或watchdog；
- production step wall不超过matched world-size SKNC的`1.5x`。

任一hard gate失败即淘汰PCUG，不扫pair数、guard阈值、candidate scale、rank、dtype、seed或projection tolerance。

## 9. Formal decision

live机制和实际evaluation adapter B8/16/32 deployment profile全过后，fresh训练`0→5`并立即跑与old134相同的
strict paired correct400。只有同时满足以下条件才允许`5→10`：

- macro5 correct`>=142`、breadth`>=6`；
- 相对old134 lost`<=8`且gained>lost；
- 至少3个suites不降，最大单task净增不超过全部正净增的`.5`；
- 每macro都有非零discordant/harmful evidence，final guard closure、rank和energy不坍缩。

首次correct`>=144`才补same/wrong/shuffled/reversed/no-video；最终成功仍要求同一single checkpoint strict
correct`>150`、correct严格优于negative controls且same-task-other至少保留correct的`.9`。若macro5未过门，
不resume、不补controls、不做seed/scale/pair sweep。

## 10. Rejected alternatives

- 不把reward tangent microbatch化继续救SRTP：这违反已封存的唯一reprofile边界，且仍在优化surrogate gradient；
- 不只增加更多success anchors：它们没有告诉当前candidate是否有害；
- 不对harmful candidate做global scalar reject/line search：一个task不能否决其余23 tasks的有益写入；
- 不直接反转harmful task的blind cotangent：paired outcomes只识别当前方向有害，没有识别完整反方向梯度；
- 不永久保存harm guards：candidate-specific冲突不是task identity；
- 不改变few-shot、rank、compiler、video encoder或negative objective，以保持本次单变量可证伪。
