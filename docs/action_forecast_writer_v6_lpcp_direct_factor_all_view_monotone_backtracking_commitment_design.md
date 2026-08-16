# V6-LPCP Direct-Factor All-View Monotone Backtracking Commitment

状态：2026-08-16 terminal non-pass，禁止full24、strict、resume、扩大backtrack或参数小扫。clean `aa819f2`首次
world1三anchor均完整exit0，但暴露
baseline measurement contract错误，不能作为科学pass/non-pass：训练前margin来自gradient-enabled CFM路径，candidate
margin来自inference CFM路径；恢复step0后fixed-action probe又把rollout B2/B1 action与batch1重算action比较。修正版只
统一这两个测量路径，不改科学变量；全量CPU=`404 passed`、architecture guard 0 hard violation。clean `202a64d`
随后完成全部三anchor与train/held/时序分析：task18在`j=5`全门通过；task9到`j=10`才接受但held仅4/8、
held/train`.18446x`；task15到`j=10`仍无共同下降candidate并正确恢复step0。故本轮科学假设终局。
简称`AV-MBC`。本轮从sealed LPCP fresh启动，完整保留MB-SOP/AR-EC的video-language carrier、matched
successful-occupancy panel、四个
correct K4 views、八个direct native-factor heads与rank16部署图，只替换raw shared gradient的有限步半径选择。

## 1. 选择依据与最早失败接口

AR-EC clean `b578d56`在task9/15/18全部复现固定outcomes、26/65/44 complete chunks与8/16/8 selected pairs。
每个view到raw等权mean的dot仍全部为正，minimum cosine=`.695/.629/.601`；最终parameter delta也与`-g`精确同向。
但同Adam候选全局L2半径写回后，三个任务都只有`1/4` correct-video views的同panel margin下降：

```text
task9  = [-.00001710, +.00000750, +.00020203, +.00006086]
task15 = [+.00027914, +.00001779, -.00001787, +.00000334]
task18 = [+.00041723, -.00028980, +.00021348, +.00021826]
```

Adam candidate radius为`.12114/.10753/.16521`，分别是raw gradient L2的`6332.5/7988.2/4293.9x`。同时AR-EC
train BA cosine/energy=`.867/.868,.765/.738,.863/.844`，held8=`.782/.779,.829/.811,.817/.811`；q/v/action、
reverse、constant和core wall全部健康。这说明optimizer方向旋转不是当前最早断点，LoRA几何也没有坏；最早接口已
缩到：**同一4/4共同一阶下降方向 -> 超出共同局部下降区间的有限步半径。**

## 2. 唯一变量与完整保留项

完整保留：

- exact language + internally ordered action-hidden K4 videos，stride5；
- 同一次真实image/language/50 Action-probe context forward、18层LPCP与DJNFR direct FactorHeads；
- 同B8双臂matched action panel、8个时间strata最大分歧state、Nmc4与四个disjoint correct K4 views；
- trajectory、view与active-task等权；source policy、normalization、split、rank16与38 targets冻结；
- AdamW LR/betas/eps/weight decay/clip和由raw gradient产生的moments/step counter；
- 每condition一次生成一套LoRA，0 teacher/target/validation/test forbidden read。

唯一变量：不再无条件采用Adam候选的完整global L2 radius，而是在同一`-g`方向上确定性回退到第一个令四个
correct-video views全部严格下降的半径。它不改变video权重、gradient方向或LoRA topology。

## 3. 确定性all-view monotone backtracking

令clip后的四view/task等权raw gradient为`g`，照常运行AdamW候选并保留optimizer state，取上界
`r_0=||d_adam||_2`。依固定顺序检验：

```text
d_j = -2^(-j) * r_0 * g / ||g||_2,  j = 0, 1, ..., 10
```

每个candidate都从相同`theta_0`写入；复用训练时已经选定的occupancy batch、trajectory IDs、四个conditioning
states与完全相同的flow times/noises。先以同一个inference-only CFM evaluator在`theta_0`计算四个exact baseline
margins；所有candidate也只用该evaluator，选择**第一个**满足
`margin_candidate_i < margin_inference_baseline_i` for all four views的`j`。gradient-enabled forward的原始margin
只保留为路径差诊断，绝不参与acceptance。不看哪个scale closed-loop更好，不选最低loss，不平均
多个candidate，也不产生多个checkpoint，所以这是一条确定性optimizer acceptance rule，不是固定scale sweep。

最多10次回退是预注册成本边界：AR-EC的方向导数与full-radius曲率反事实预计转折落在约`1/64--1/256`，
`1/1024`再留4倍半径余量。若仍无共同下降，恢复`theta_0`并作为科学non-pass封存；不继续缩小、改LR或挑view。
若accepted step在native q/v/action或fixed-action response上消失，也按门终局。

回退不增加environment step，不重新运行video backbone，不重选occupancy，只增加冻结policy的CFM margin forwards。
fixed-action response的step0与post都使用同一retained query、noise、batch1和policy forward；不再把动态rollout
batch的stored action作为baseline。它只增加一次必要的batch1 policy forward。
每个trial及其四个margin、wall、accepted index和实际native response都必须记录，后续依据真实耗时决定formal实现是否
需要等价的batch/kernel优化；不能删减四view acceptance来换吞吐。

## 4. Canonical实现边界

- 原位替换AR-EC config/checkpoint/completion/evaluator schema；不保留runtime strategy switch；
- `reward_gradient_update.py`唯一拥有Adam upper radius、candidate写入、first-acceptable backtracking和最终parameter
  commitment；
- `reward_cycle.py`只提供已经存在的四view conditioning/panel，并把accepted evidence写入cycle metrics；
- `aa819f2`三run roots完整保留为工程诊断，禁止resume或据其search rejection裁决科学假设；修复后必须从新的
  clean pushed commit与fresh roots运行；
- 首轮只授权world1 task9/15/18机制实现；formal保持blocked。三anchor全过后才把同一acceptance扩展为对每个
  active task×4 views共同成立的distributed full24合同；不能只检查每rank首个probe；
- fresh incompatible，不加载AR-EC heads或optimizer；AR-EC由Git、三run roots和terminal artifact保存。

## 5. 固定三anchor机制门

task9/15/18全部满足才允许实现formal full24：

1. outcomes=`2/1,2/0,1/2`，complete=`26/65/44`，selected=`8/16/8`，matched B8与0禁读不变；
2. raw four-view shared descent coverage=`4/4`，每个dot positive/finite；
3. 先以同一inference evaluator测一次`theta_0`，再严格按`j=0..10`搜索；candidate都从同一`theta_0`写入，只
   接受first all-view-monotone candidate，并记录gradient-path到inference-baseline的四项offset；
4. 三任务都在不超过10次回退内accepted，四个margin均严格下降；
5. Adam moments保留，final到`-g` cosine至少`.999999`，实际radius等于`2^-j r_0`、relative error不超过`1e-6`；
6. 八head、q/v/action native BA与同batch1路径的fixed-action response非零；若恢复step0，二者都必须严格为零；
7. train four-view BA cosine/energy至少`.40/.55`；validation8 aggregate至少`.30/.48`、6/8 tasks过`.15/.40`，
   raw factor cosine至少`.30`、action cosine至少`.15`、held/train L2至少`.30x`；
8. reversed BA relative-L2至少`.50`，constant/natural不超过`.005`；
9. 记录search trial数与CFM wall，0 OOM/nonfinite；若机制虽过但成本不能扩展，先做等价吞吐优化，不弱化acceptance。

任一科学门失败即终局，不做LR、max-backtrack、halving factor、rank、scale、seed或view小扫。

## 6. Full24与真实性能裁决

三anchor全过后才补全distributed active-task acceptance，并从sealed LPCP fresh运行full24 cycle1。全局shared proposal
必须对所有active task×4 correct-video panels共同单调；不能每task各自生成一套LoRA或参数delta。cycle1后立即K4
strict paired400，继续使用correct至少142、breadth至少7、相对LPCP lost不超过15且gained不少于lost的继续门。
稳定资格仍需相邻两个single checkpoints均至少142、均值至少145、churn不超过20、Jaccard至少`.85`、final lost
不超过10；首次约145且retention过门立即补same-task-other、wrong、shuffled、reversed与no-video。

若AV-MBC能让训练panel共同下降但strict仍换手或下降，最早接口后移到matched successful-occupancy preference与真实
held closed-loop reward的对齐，而不是再改optimizer、video carrier或LoRA健康度。

## 7. 负结果边界

本轮只检验“MB-SOP credit + raw shared direction + Adam upper radius + deterministic all-view monotone backtracking +
one fresh cycle”。负结果不否定memory token、rank8、few-shot、生成LoRA、其它trust-region形式或未来生成LoRA后的
task-local RL。
