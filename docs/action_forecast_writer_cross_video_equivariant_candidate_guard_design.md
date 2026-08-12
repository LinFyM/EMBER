# Cross-Video Equivariant Candidate Guard

状态：2026-08-12从NPCG strict non-pass后的最早失效接口选定，当前唯一active successor authority；尚未实现、
profile或训练。实现前的科学变量、采样、机制门和formal裁决在本文封存，不能根据live结果增加companion数量、
放宽约束或改ridge/scale/threshold。

## 1. Latest evidence and earliest failure

Negative-Preserving Candidate Guard已经证明以下接口可用：task-addressable work queue、train24等权B20 blind
functional cotangent、actual base/candidate K2x2 pairing、first-stable/harmful correct guard、negative-preserving
affine correction、FP32 Program、native 38-target rank16 compiler以及Program→LoRA→effective BA→action传递。
五个formal macros的constraint closure、rank和energy持续健康，但macro5 single-checkpoint strict paired400只有
`135/400`、breadth5，相对old134 retained/gained/lost=`117/18/17`、churn35。

train24x50 action-hidden cache把最早失败进一步定位到condition neighborhood：16个first-stable task的单点key
到同任务其它videos的cosine均值/中位数/p10=`.90100/.91979/.80615`，但正交残差均值/p90仍为
`.40954/.59171`；合并五宏全部observed stable rows后平均残差仍约`.27650`。因此精确保护一个训练video点，
不能定义同任务其它video或held rollout occupancy上的policy-effective support。

本设计只改变这个接口：每task每macro增加一条**训练期、action-hidden、同任务、保持内部顺序**的companion
video，并令每个shared Program update对primary/companion的condition差分为零。部署仍严格读取一条video并生成
一套LoRA；没有few-shot set encoder、video/feature/LoRA平均、best-video选择或第二套adapter。

## 2. Single causal variable

每task的primary correct feature为`f_i in R^256`，companion feature为`f'_i`，定义：

```text
e_i = f'_i - f_i
E   = stack(e_0, ..., e_23)
```

blind update不再只位于persisted success keys `A`的nullspace，而位于`[A; E]`的nullspace：

```text
D0 = argmin_D ||F D + Y||_F^2 + ||N D||_F^2 + lambda ||D||_F^2
     subject to A D = 0
                E D = 0
```

`Y`仍是每task B20跨episode source-action functional cotangent，`N`仍是24个wrong/shuffled/reversed
zero-response rows。`E D=0`不要求两条video的完整历史v6 LoRA相同；它只要求本macro新增的shared Program motion
相同，从而不让demo-specific nuisance决定新写入方向。

NPCG final correction同时保护negative response与equivariance response。令`G`为persisted first-stable及当前
stable/harmful primary correct guards，`P=[N; E]`：

```text
C* = argmin_C ||C||_F^2
     subject to P C = 0
                G (D0 + C) = 0
D1 = D0 + C*
```

于是`N D1=N D0`、`E D1=E D0=0`、`G D1=0`。final reward guard既不能重新打开negative-video response，
也不能重新引入primary/companion不同的新增motion。实现复用同一个small feature-space affine solver，把其输入从
`negative rows`概括为`response-preserving rows`；不新增parallel trainer/module或兼容路径。

## 3. Why one companion is the sealed choice

对retained train24x50 cache做了与canonical `TeacherVideoSchedule(seed=20260722)`一致的50-panel离线审计。
primary为visit `t`，companion取同一no-replacement cycle中primary后的第一个action-query-disjoint demo：

- 单companion的`E`在50个panels均rank24；投影`Null(E)`后correct feature energy的
  mean/median/p10/p90/min=`.77415/.78016/.69457/.84611/.51833`；
- correct-minus-reversed过程差分能量保留mean/median/p10/p90/min=
  `.78444/.78943/.70238/.85881/.47094`，说明约束没有结构性删除有向时序证据；
- 一对video只覆盖部分nuisance：其它held same-task差分在`span(E)`外的残差比例mean=`.76008`。它不是声称
  一次就消除全部video变化，而是每macro重采一对，以Monte Carlo方式约束shared update；
- 在NPCG macro5 Program geometry中，单companion把held video-difference Program response从mean
  `.0003000`降到`.0002261`（ratio`.7536`），同时reverse Program response fraction的mean/median=
  `1.035/.966`；
- 1/2/3/4 companions的correct energy约为`.774/.639/.529/.436`。更多companions继续降低held nuisance，
  但快速压缩可用condition energy并改变计算量；历史uniform compression已有support损伤证据。

因此首个可证伪版本固定**恰好一条**companion，不做K sweep。companion不进入checkpoint bank；每macro按sealed
schedule重采，避免把全部video differences永久堆入低维Program rowspace。

## 4. Sampling and information wall

- primary schedule保持NPCG one-shot `TeacherVideoSchedule(seed=20260722)`逐task逐visit完全不变；
- companion来自同一个50-demo no-replacement cycle中primary之后的第一个合法demo，依次跳过primary和本macro
  B20 action-query episodes；不使用新随机seed、outcome、长度、task ID route或feature相似度挑选；
- primary和companion各自保持原始帧顺序、stride5并只读RGB；两者都不读teacher action、proprio、reward、
  terminal、filename、pose或hidden normalization；
- companion只构造`E`，不生成functional target、不参与paired rollout、不进入deployment、不持久化；
- actual K2x2 candidate pairing继续只测primary condition生成的完整candidate LoRA；
- correct与shuffled/reversed的区别仍依赖同一primary内部的有向过程；wrong仍替换video而保持target language。

language-only/static bypass不能满足合同：condition feature仍是same-language actual-image减zero-image的ordered
policy innovation；`N D≈0`要求wrong/shuffled/reversed无新增response，而`E D=0`只消除同任务不同正确demo之间的
nuisance，不允许语言直接产生独立LoRA。

## 5. Multi-task coexistence and inherited advantages

- 保留历史143的frozen v6 semantic/procedure graph和task-complete recipe，不重写encoder/compiler；
- 保留PICK-GC已验证的ordered goal-causal policy innovation key和condition-local FP32 Program；
- 保留Work-Queue的completion-driven task claim，full24仍按task ordinal等权汇合；
- 保留actual candidate pairing，不用functional loss、key cosine或内部margin替代closed-loop harm；
- 保留NPCG affine correction，因此correct reward protection、negative suppression和cross-video equivariance在
  同一个shared update内可组合；
- 保留native完整38-target rank16 factor heads，不做compression、expert route、checkpoint union或LoRA平均。

与K4不同，本设计没有部署时多video集合输入，也不提取/平均四条video的共同表示。它是在训练共享condition→
Program map时施加同任务video-pair增量等变约束，canonical rollout仍是one-shot。训练额外video compute会在
profile中计入真实wall，不用matched one-shot以外的FLOPs优势作结论。

## 6. Canonical implementation boundary

1. `TeacherVideoSchedule`原位支持一个training companion，并让cost order计入primary+companion sampled frames；
2. `generate_condition_graph`在同一个task graph构造期额外编码companion feature，历史v6 base slots、primary
   correct LoRA和negative construction不变；
3. full24 gather携带`correct/negative/companion/cotangent`，差分`E`只在shared rank按ordinal形成；
4. blind solver把`E`与persisted keys共同作为zero-motion anchors；final affine solver保护`[N;E]` response；
5. profile显式报告`E` rank、correct/reverse retained energy、blind/final equivariance motion、negative和
   equivariance correction closure，以及原full48 rank/energy/action evidence；
6. fresh-incompatible config/schema/checkpoint/eval family替换canonical owner；sealed NPCG config和artifacts只作
   历史证据，不能exact-resume；
7. fresh training允许同节点world size `1--6`，有多少合适卡就用多少、至多6；不等待凑卡。低利用率小占用卡有
   足够峰值余量且不会明显互相干扰时可共享。

没有新worker pool、preload、hash、逐tensor scan、重复forward、dtype扩展或防御性fallback。companion本来就是
科学变量所需的唯一额外video forward。

## 7. Fast falsification

### 7.1 CPU and offline gates

- primary demo schedule与sealed NPCG逐visit完全相同；companion必须与primary及B20 action episodes不同，且
  exact-resume按task visit得到同一pair；
- synthetic solve满足`E D0≈0`、`E(D1-D0)≈0`、`N(D1-D0)≈0`、`G D1≈0`；无guard时不改变blind update；
- final correction等价于response-preserving minimum-norm solve，dependent rows按同一数值rank处理；
- single-companion cached `E` rank24，correct retained energy median`>=.65`、reverse-process retained energy
  median`>=.70`；
- checkpoint不保存companion video/feature或第二套LoRA，旧NPCG checkpoint必须fresh-incompatible。

### 7.2 One discarded full24 macro0

从historical v6-fast、zero Program、empty bank运行一个live mechanism profile：

- 24 primary与24 companion rows齐全、逐task distinct且action-query-disjoint；`rank(E)=24`；
- blind及final `E` motion RMS / primary motion RMS均`<=1e-5`；`E(D1-D0)`closure`<=1e-5`；
- `N(D1-D0)`closure`<=1e-5`，final negative/unprotected ratio`<=.15`，wrong/shuffled/reversed各至少6/8 tasks；
- full48 projected analysis rank`>=24`、final/blind energy`>=.25`、alignment为正；
- exact 48 paired states完成，至少2个discordance、至少1个gain或loss、stable-success至少6 tasks，使actual
  candidate evidence有内容但不预选优劣方向；
- protected Program/LoRA/effective-BA/fixed-action closure通过，四suite有非零unprotected action response；
- total wall相对world-size scaled SKNC baseline`<=1.5x`，无OOM/nonfinite。

任一项失败即淘汰当前CVEG合同，不增加companions、不扫seed/ridge/scale/SVD tolerance/threshold/rank/dtype，
不靠放宽negative或energy门救结果。工程合同违约只允许一次精确复现的原位窄修。

## 8. Formal decision and paired400 timing

profile全过后直接做deployment longest-video batch profile，只按真实LoRA/s和OOM平台选择batch；随后从v6-fast
fresh运行`0→5`，每macro保存并报告bank、rank、energy、negative/equivariance closure、paired outcome churn与
task wall。macro5完成后立即做single-checkpoint strict paired correct400，不用loss或内部几何挑checkpoint。

只有同时满足以下门才exact-resume `5→10`：

- correct`>=142/400`、breadth`>=6`；
- 相对immutable old134 lost`<=8`且gained`>lost`；
- 至少3个suites不下降，任一单task不得贡献超过总net gain的一半；
- 五宏negative/equivariance closure、rank、energy和paired evidence持续健康。

若macro5 correct`<142`、lost`>8`或出现新的单task换手集中，CVEG直接formal non-pass，不先跑controls或小修。
若续到macro10，立即再次strict paired400；只有correct至少达到历史143附近且相对old134稳定，才运行same/wrong/
shuffled/reversed/no-video controls。最终成功仍要求同一single checkpoint strict correct严格`>150/400`，并证明
correct video沿有用policy direction优于controls；内部equivariance本身不能宣告方法有效。

## 9. Rejected alternatives

- 不把companion用于生成第二套LoRA或做平均：这会改变one-shot部署并掩盖单video鲁棒性；
- 不把companion当额外functional target：同任务恒定target会继续允许task-identity shortcut；
- 不存更多success point keys或key均值：NPCG formal已经否定point closure足以定义held neighborhood；
- 不一次用2--4个companions：离线审计显示可用condition energy随K快速下降，且第一步应保持单变量与吞吐；
- 不强制完整primary/companion LoRA相同：历史v6 base Writer本身允许有用的视角/状态适配，本设计只约束新增
  shared write不跟随demo nuisance；
- 不删除ordered negative或降低negative weight：正确顺序仍是任务过程知识的结构证据；
- 不用held outcome、task expert route或same-video score选择companion。
