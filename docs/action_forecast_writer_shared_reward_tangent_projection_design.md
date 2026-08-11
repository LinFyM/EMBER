# Shared Reward-Tangent Projection

状态：2026-08-12 active implementation authority。canonical实现与fresh-incompatible config已完成；constant-
memory landmark、K4 credit、small-dual projection、checkpoint隔离及完整CPU回归`358 passed`。尚无GPU、checkpoint
或closed-loop成绩。只有discarded live和deployment门依次通过并由clean pushed commit封存后，才允许formal
fresh训练。

SRTP从SKNC macro5的正式终局出发：保留historical v6-fast Writer、PICK-GC ordered goal-causal condition key、
B20 blind source-action proposal、FP32 Program、SKNC all-success anchor bank和native 38-target rank16 compiler；
只把最终24-task shared Program write从“仅受零阶success-key equality保护”改成“同时满足当前mixed-outcome
on-policy reward tangents的共享半空间”。reward约束直接作用在task汇合后的最终memory update，不再先约束
task-local proposal再期待full48 solve保持它。

## 1. Formal result that selects this interface

SKNC fresh`0→5`的macro5仍有15个persisted anchor tasks、projected rank36、energy ratio`.59195`和
protected/unprotected Program ratio=`1.276e-7`；Program→LoRA→action closure没有失效。strict paired400却只有
`137/400`、breadth7，相对old134为`121 retained/16 gained/13 lost`、churn29。Long task1净`+7`和Goal6净`+2`
伴随两个Object tasks净`-5`、Spatial净`-1`。所以：

1. blind B20仍有可用增益，不应整条丢弃；
2. 单条train video的all-success condition key不是held video/initialization support坐标；
3. 最终shared write需要真实on-policy一阶方向约束，而不仅是某些key完全不动；
4. 新约束必须在24 tasks汇合后生效，否则会重复task-local cotangent被shared map改变的问题。

历史Reward-Credit已经证明K4 mixed outcomes能产生finite、nonzero的Program cotangent；同源profile有11个mixed
tasks、四suite覆盖、60/96 successes。其旧full-trajectory Nmc4实现处理4452 replay chunks和928 functional
forwards；连续tangent直接写native BF16 factors时约`1e-8 RMS`，没有足够物理幅度。OSG-PC随后用每条success
trajectory全prefix VJP保护B20，但唯一profile发生rank-local长尾和NCCL watchdog，未到shared mechanism report。

SRTP不恢复这两条已退役执行图：reward tangent不直接替换B20幅度，也不回放完整trajectory；它只判断最终shared
B20 update是否沿mixed-task reward的一阶有益半空间，并用常数大小occupancy landmarks计算该约束。

## 2. Single causal variable

```text
SKNC:
  blind B20 cotangents + negative zeros
  -> all-success anchor-null full48 shared update D0

SRTP:
  the identical D0
  + current mixed-outcome on-policy reward tangents
  -> nearest anchor-null shared update D* satisfying every reward half-space
```

不改变rank、scale、seed、teacher-video数量、PICK key、source functional objective、negative schedule、split、
normalization、source policy、LoRA topology或deployment graph。固定landmark budget只是使这一个reward-tangent
变量具有可接受wall和显存，不是第二个scientific objective。

## 3. One-shot graph and information wall

每个macro、每个train task仍读取exact language和一条correct ordered action-hidden video：

```text
language + one video
  -> frozen v6 evidence
  -> PICK goal/causal feature phi_i
  -> current FP32 Program
  -> one complete native rank16 LoRA

B20 same-task cross-episode source action queries
  -> blind Program cotangent and SKNC shared update D0

same LoRA + K4 official random-reset train rollouts
  -> four binary outcomes
  -> mixed panels only: fixed occupancy landmarks + LOO signed CFM cotangent r_i

D0, anchors, {(phi_i, r_i)}
  -> one shared convex projection D*
  -> FP32 Program add
```

teacher action、teacher proprio/state、teacher reward/terminal、task ID、filename、pose和hidden normalization始终
不可读。训练rollout的observations、policy-generated action chunks和binary environment outcome只进入train24
ephemeral credit；不成为Writer输入、不写checkpoint、不用于validation/test。deployment仍在rollout前运行一次
Writer，随后不再看video、reward或Program bank。

## 4. Fixed-budget mixed-outcome reward tangent

### 4.1 K4 outcome semantics

对同一correct-video LoRA执行四个独立random-reset episodes，令`y_e in {0,1}`。只有`0 < sum y_e < 4`时计算
reward tangent；LOO advantage固定为：

```text
a_e = (4 y_e - sum_j y_j) / 3
```

每episode先等权，再在K4内等权。success trajectory被正向模仿，failure trajectory被反向去模仿；全成功与
全失败panel的advantage都严格为0。全成功task继续由SKNC current/persisted key equality保护，全失败task继续
使用blind B20 acquisition proposal。reward不提供可调scale、margin或accept/reject gate。

### 4.2 Four occupancy landmarks per episode

rollout期间每episode最多只保留四个replan rows：first、last和两个seeded uniform-reservoir interior rows；结束后
按真实replan ordinal排序。少于四个unique rows时保留全部并按该episode实际row数归一。每row只保留一次CFM
replay所需的policy observation、生成action chunk、executed-prefix length和seed identity。

这不是teacher trajectory cloning。landmarks来自当前policy在不同初始化上的on-policy occupancy；first/last保证
任务边界，两个interior reservoir rows无偏覆盖中间阶段。所有未选rows立即释放，mixed credit完成后四条episode
buffers全部释放。历史同源panel的928 forwards因此固定降为每mixed task `Nmc4 × one B16 forward = 4`，11个
mixed tasks时为44，而不是随Long horizon增长。

沿用封存的Nmc4 keyed Beta(1.5,1) time和Gaussian noise语义；完整logical `<=16` rows先生成MC panel，再按
physical batch切片。batch不得改变seed或landmark identity，不用batch1、重复forward、扩dtype或内容hash。

### 4.3 Program cotangent

对task `i`的selected rows定义episode-equal signed executed-prefix CFM objective：

```text
J_i = (1/4) sum_e a_e mean_{landmarks in e} mean_{mc=1..4} L_CFM
r_i = grad_{H_i} J_i,          H_i = phi_i M
```

mixed task的`r_i`必须finite且nonzero；homogeneous task严格zero且不做functional forward。`r_i`只定义一阶
half-space，不直接作为memory delta，因此避免历史Reward-Credit连续tangent低于native factor ULP的问题。

## 5. Projection on the final shared update

Program memory `M`形状为`[F,320,256]`，`F=256`。SKNC先按原合同产生已经满足anchor equality的shared update
`D0`。令anchor rows为`A`，FP64解析projector为：

```text
P_A = I - A^T (A A^T)^dagger A
```

对每个mixed task，最终更新的一阶reward change为：

```text
c_i(D) = <r_i, phi_i D>
```

SRTP求唯一Euclidean projection：

```text
D* = argmin_D 0.5 ||D - D0||_F^2
     subject to A D = 0
                c_i(D) <= 0 for every mixed task i
```

`D=0`始终可行。因为`D0`已在`null(A)`，每条约束在该子空间的normal是outer product：

```text
u_i = P_A phi_i
q_i = u_i^T outer r_i
b_i = c_i(D0)
G_ij = <u_i,u_j> <r_i,r_j>
D* = D0 - sum_i lambda_i q_i
```

dual是最多24维nonnegative convex least squares。只把`G,b`传CPU，以FP64 eigenspace和已有SciPy NNLS求
`lambda>=0`；零特征值按machine-epsilon numerical rank处理，不引入scientific tolerance、iteration sweep或
reward scale。大tensor correction在GPU FP32一次合成，constraint诊断用FP32 GEMM，不继承TF32低位误差。

解析性质：

- 没有mixed task或所有`b_i<=0`时，`D*`逐元素等于SKNC `D0`；
- constraint正尺度、task permutation、重复或线性相关rows不改变投影；
- 所有final `c_i(D*)<=0`到FP64/FP32 numerical tolerance；
- `A D*=0`，所以已有all-success Program/LoRA/action closure保持；
- 若`D*!=0`，projection theorem给出`<D0,D*> >= ||D*||^2 > 0`，不会把shared blind direction整体翻转；
- 若唯一可行点接近0，方法按机制门non-pass，不用scale或删task constraint强推。

negative correct/shuffled/reversed/wrong features仍以SKNC同一full48 zero RHS参与`D0`，SRTP不新增negative rollout。
投影后必须重新报告negative motion；若reward correction破坏negative null门，说明当前shared坐标不能同时容纳
causal selectivity与reward credit，直接否决。

## 6. Why task knowledge, order and video are necessary

SRTP继续使用PICK-GC：

```text
phi(video) = normalize([
  terminal-quartile policy innovation - whole-video mean,
  centered causal-prefix policy innovation
])
```

terminal goal residual提取初态到目标态变化，causal prefix保留动作阶段的有向顺序。reverse交换有向端点并翻转
prefix，shuffle破坏连续阶段，wrong改变对象/关系过程。zero-image subtraction使language没有condition value；
static/no-video给zero dynamic feature。reward half-space只在correct ordered `phi_i`及其真实K4 occupancy处定义，
counterfactual features仍要求incremental zero。若video被忽略或correct/negative key碰撞，raw/full48 rank、
condition、negative motion或reward projection gate必须失败。

K4是四个初始化，不是four-shot videos。reward只认可跨reset真实完成所支持的policy direction，不能监督复制teacher
的速度、路径、视角或抓取角度。训练每macro重采teacher video和K4 states，shared memory必须把多个task/video
访问映射到同一组满足所有current reward constraints的参数，而不是保存per-task checkpoint或expert route。

## 7. Coexistence argument and limits

SRTP在同一个shared solve中区分三类train evidence：

1. all-success：current/persisted condition key保持完整Program零运动；
2. mixed：最终shared update必须降低或不变其signed on-policy reward objective；
3. all-failure：不伪造reward方向，保留B20 acquisition proposal。

这比SKNC的point key包含occupancy一阶信息，比OSG-PC的task-local per-success cone更晚地约束真实shared write，
也比直接Reward-Credit保留更大的B20/native rank16物理写入。它仍只读train24 reward；train occupancy不必然外推
held tasks，CFM tangent也不等于return gradient，所以真实性能只能由macro5 strict paired400裁决。

## 8. Rejected alternatives for this iteration

- 不直接用reward cotangent替换B20：历史cycle1物理写出过小且134/400、14/14换手。
- 不恢复OSG full-prefix/per-success VJP：它随horizon增长并已触发watchdog；SRTP固定16 rows/task且每task一个
  signed tangent。
- 不只增加更多SKNC anchors：macro5容量健康但held lost13，继续扫first/K/threshold没有新credit信息。
- 不恢复RLS、rank14、expert bank或language route：它们分别保护offline rows、损伤base、缺held support或形成
  bypass。
- 不同时改few-shot deployment、rank、scale、seed、MC estimator或negative arm；这些是独立问题。

## 9. Implementation ownership

实现必须原位替换唯一SKNC active runtime，不保留strategy flag或第二trainer：

- `expert_manifold/v6_prior_training.py`继续拥有task-complete graph、K4、gather、shared solve和checkpoint；
- 新的单一reward-tangent owner只负责fixed landmark credit与small dual projection；不恢复旧full-replay APIs；
- `reward/rollout.py`只保留constant-memory landmark collector，task结束释放；
- `writer/condition_update.py`仍是唯一shared memory solver owner，新增post-SKNC shared projection而非第二full48
  implementation；
- canonical config/checkpoint/adapter升fresh-incompatible SRTP schema；SKNC config保持
  `formal_result_sealed`并拒绝resume；
- deployment Writer graph与B32 generation path不变，但新family仍须做实际adapter smoke并封存。

历史Reward results/gate解析可读保留；旧Reward/OSG executable只从Git选择性移植所需公式，不恢复整个路径。

## 10. Fast falsification gates

### 10.1 CPU and synthetic

1. landmark collector每episode最多4 rows，first/last存在、interior reservoir可复现，未选rows不保留；
2. K4 LOO advantage、episode-equal weighting、homogeneous exact-zero和mixed nonzero cotangent；
3. no-constraint/raw-feasible逐元素退化SKNC；单/多/相关/重复constraint的NNLS解、KKT与permutation invariance；
4. final shared reward motion全部非正，anchor Program/LoRA/fixed-action motion保持zero，nonzero projection与D0正对齐；
5. full48 negative RHS仍zero，projection前后negative motion、rank与energy直接报告；
6. teacher/validation/test forbidden reads为0，policy-generated replay不进checkpoint；
7. checkpoint保存Program、anchor bank、cursor/sampler/RNG/topology，fresh SRTP不能读SKNC checkpoint；
8. 一个Writer、trainer、shared solver和deployment path，无旧full-replay fallback。

### 10.2 One discarded live macro

从historical v6-fast、zero Program、empty bank执行一个完整train24 macro：24 videos、480 B20 queries、96 K4
rollouts。实际launch前同时live检查gpu01/gpu02，选同节点至多6张有益A40；少量显存占用不自动排除，world size
不必为6。exact command、quota、peak、NUMA/NCCL和fresh root按formal-launch流程封存。

必须同时满足：

- 四suite task/outcome完整；mixed tasks`>=8`且至少覆盖3个suite，all-success tasks`>=6`且四suite都有；
- 每episode retained rows`<=4`，mixed functional forwards恰为`4 * mixed_tasks`，无full-prefix replay；
- mixed cotangentsfinite/nonzero，homogeneous cotangent/forward exact zero；
- blind shared update至少2条reward constraint为正且projection实际改变，否则SRTP退化为SKNC；
- final reward violations为0，anchor closure过门，projection/D0 energy ratio`>=.25`且alignment为正；
- projected feature rank`>=24`、negative/correct motion ratio`<=.15`，Program/LoRA/fixed-action响应非零；
- 0 OOM/nonfinite/watchdog/forbidden read；step wall不超过matched SKNC按world size缩放值的`1.25x`；
- throughput用稳定最大B20/replay batch和原生BF16/TF32，不为低位一致降速。

任一hard gate失败即拒绝当前SRTP，不扫landmark数、Nmc、reward scale、QP tolerance、anchor threshold、rank、
dtype或seed。

这里的mixed suite门在GPU launch前按既有封存证据纠正过一次：同一env/policy seed的historical Reward profile为
`5/3/2/1`，matched SKNC world3 profile为`4/3/2/0`，差异只来自一个Long task在`3/4`与`4/4`成功边界上随正常
batch低位变化切换。要求单个stochastic panel四suite都mixed会把已知的允许低位差异变成必然/偶然hard fail。
因此不挑seed、不改schedule，保留mixed总数`>=8`，要求至少3 suite有一阶tangent，同时用四suite all-success
anchor coverage和完整24-task negative/B20路径覆盖第四suite。flow credit root仍复用封存Reward的`2026081103`。

### 10.3 Deployment and formal

live gate通过后，用实际evaluation adapter在longest-first val8x4 correct panel测B8/16/32并选择最高stable
LoRAs/s；不重复single forward。随后从zero Program/empty bank fresh`0→5`，每macro重采video/K4/landmarks，
macro5立即跑与old134完全相同的strict paired correct400。

只有同时满足以下条件才允许exact-resume`5→10`：

- correct`>=140`、breadth`>=6`；
- 相对old134 lost`<=8`且gained>lost；
- 至少两个suite不降，最大单task正净贡献占全部正净贡献`<=.5`；
- reward constraint active且final可行，anchor/negative/Program→LoRA→action closure不坍缩。

首次correct`>=144`才补same/wrong/shuffled/reversed/no-video；macro10成功门仍是strict correct`>150`。最终
checkpoint必须correct严格优于各negative、same-task-other/correct`>=.9`，且仍是一个shared method、一个
single checkpoint。未过macro5门就停止，不靠训练长度或小参数sweep补救。

## 11. Interpretation boundaries

- projection退化SKNC：B20与本reward tangent没有可观察冲突，SRTP没有新增机制；
- projection几乎归零：当前Program坐标中的acquisition与reward constraints不可共存；
- train reward constraints满足但strict lost仍高：train mixed occupancy仍不能外推held support；
- lost降到门内但absolute不增：support alignment有效，下一接口是all-failure task的reward-guided acquisition；
- correct提高但controls同步提高：video causal key没有提供独有知识，方法科学non-pass。

负结果只淘汰“PICK/SKNC + fixed-landmark mixed reward tangent + final shared projection”这一组合，不否定所有
on-policy credit、few-shot、task-level manifold supervision或其它跨video support representation。
