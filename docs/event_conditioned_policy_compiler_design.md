# EMBER-ECP: Event-Conditioned Policy Compiler

状态：2026-08-23 **mobile-rank4 residual容量门已判为mixed；没有active successor Writer或solver**。
旧ECP Stage 1 v1--v24、MDCO、PECS及本轮oracle的精确实现、结果和退出原因由`docs/research_history.md`、Git与formal
artifacts保存；最新裁决为
`docs/evidence/ecp_20260823/ecp_mobile_rank4_residual_capacity_gate_20260823.json`。

## 1. 当前裁决

ECP的核心目标与Stage 0候选表示尚未被证伪，但以下实现家族已经有足够反证，不再恢复或做小变体：

- deterministic mean `q_pi`；
- learned Program-to-A/B hyperdecoder；
- 19或90个mapping上的21M target/rank query compiler；
- v24、MDCO以及width/rank/head/fusion/LR/seed后继；
- 只在教学视频稀疏support frames上拟合exact policy effects的PECS；
- 当前48-state三particle effect bank、stable fixed-A carrier与12-step Delta-B realization的组合；
- 用open-loop geometry、own retrieval或LoRA cosine替代早期closed loop。

本轮privileged问题已经得到否定答案：

> 在真实闭环状态分布上，用多个独立成功策略定义一个分布式策略等价类后，固定、受约束的rank16实现器能否从稳定
> carrier出发，恢复显著且广泛的task-local闭环能力？

结果由carrier `43/250`提高到`78/250`，但breadth仅3/5、Goal/Long为0、oracle recovery `.304`，整体Realization
non-pass。该轮证据把断点收窄到realization，但当时尚未分离fixed-A capacity与effect objective/calibration。复盘形成新的强科学卡
后，把三个既有成功members解析投影到carrier-A行空间并完成strict paired750：三个arms只保留`67/295=22.71%` matched
direct successes，Goal/Long共300行全0。因此fixed-A已被分离为行为上binding的参数约束并退出主线；effect objective在允许
row-space移动后是否仍不足尚未回答。不训练video predictor、shared compiler或Writer联合模型，也不扫solver、rank或插值系数。

## 2. 方法总图

最终deployment仍是：

```text
exact language + K条action-hidden有序视频
    -> frozen PI0.5-native observer
    -> ordered event/layer/family Program
    -> policy-effect distribution
    -> fixed differentiable realization solver
    -> one complete 38-target rank16 LoRA
    -> frozen source PI0.5 closed loop
```

privileged Stage 1A/1B中的task action、policy member、rollout observation、reward、BDDL progress和simulator state只用于
学习或裁决共享实现机制，不是deployment输入。最终Writer在rollout前运行一次，仍只读取语言与视频。

当前LIBERO资产没有证明same-endpoint/different-required-process的授权task pair。因此，在补充经审计的
process-identifying source-unseen meta tasks前，方法claim严格收窄为`video-conditioned scene/goal/order adaptation`，
不声称通用process understanding。

## 3. Stage 0：candidate observer，而非已证明的process encoder

保留native v3 checkpoint作为candidate。每个stride-5视频帧使用exact language、真实图像prefix、固定Gaussian
`[50,32]` suffix和原生Action Expert图，得到：

```text
Z_owner[k,t,o,h,d] : [K,T_k,38,50,128]
Z_patch[k,t,p,d]   : [K,T_k,256,128]
Z_lang[k,t,l,d]    : [K,T_k,L,128]
```

有序event segmenter提供最多`E=8`个slot。`E`是固定容量，active slot数由内容学习；视频段落与slot的对应关系不是手工
规定。每条视频内部保序，videos只在event-aligned集合层置换不变聚合，得到：

```text
P_process[e,o,d] : [8,38,128]
rho[e]           : [8]             event presence
sigma[e,o,d]     : [8,38,128]      cross-video uncertainty
```

native v3只通过non-degeneracy/task separation，尚未通过完整process identification。进入最终Writer前必须补齐只读机制门：

- full video相对first+final；
- same language/goal下不同scene；
- same procedure下不同object；
- probe replacement stability；
- event boundary与action/contact阶段对应；
- 若未来获得合法数据，再加入same endpoint/different required process。

shuffled/reversed不进入训练、loss或checkpoint选择。中段重排最多作为observer只读定位，不可选择最终模型；正式时序
特异性仍只在最终候选checkpoint冻结后做严格配对closed-loop control。

历史Action Meta-LoRA matched arm已完成且效果中性。它保留为control，不是默认observer authority；新路线不自动加载。
只有新的matched证据显示它对probe稳定性、视频因果性或闭环有明确净收益时才重新启用并冻结。

## 4. Stage 1A：successful-policy equivalence identification

Stage 1A不生成LoRA。它建立一个有显式member轴、state-support轴和stage轴的privileged teacher bank。

### 4.1 独立成功策略

现有earliest/latest来自同一优化轨迹，只能算两个checkpoint particles，不能冒充两个真正独立policy lineages。首个强
oracle至少使用：

- 现有step2000 task expert；
- 一个不同seed、独立优化轨迹、固定step2000的新rank16 task expert；
- 现有earliest successful checkpoint作为第三个辅助particle。

新member只有在预注册fixed-state closed loop上至少产生一条strict success并保留完整occupancy时才进入successful
particle bank；不按task挑checkpoint或融合LoRA。

### 4.2 Occupancy-complete state support

每个task的首个formal bank固定为四类等权support：

```text
S = S_initial U S_successful U S_candidate U S_recovery
```

- `S_initial`：official fixed init IDs `0,7,14,21,28,35,42,49`，共8个；
- `S_successful`：三个member各自一条strict-success trajectory的8个等时间/进度strata，共24个；
- `S_candidate`：上一条最强PECS trajectory candidate在fixed init ID 0上的8个strata；
- `S_recovery`：source policy按fixed序列`1,26,2,27`取得的第一条failed trajectory的8个strata。

若某candidate轨迹提前终止，按实际replan位置做8-strata无重复采样；不足8个replans是数据合同失败，不用重复帧填充。
四类先等权，再在类内等权，避免Long的帧数支配task或stage权重。每个anchor保留官方双相机+state-conditioned language
prompt形成的preprocessed policy observation、固定noise seed、归一化stage、BDDL progress、生成policy与最终success。

`S_recovery`只为privileged realization补全闭环支持；它不能写入deployment Program。后续共享`q_V`只能预测视频可识别的
task/scene/order effect distribution，rollout-only recovery信息只能进入task-independent prior、uncertainty或共享solver。

### 4.3 Policy-effect particles

在每个真实policy observation上，source、carrier和每个successful member使用完全相同的fixed antithetic noise运行官方
10-step flow integration，保留：

```text
owner[m,s,o,4,128]       # t=1 native all-layer owner lattice的DCT4
flow[m,s,10,50,32]       # 完整denoising velocity
action[m,s,10,50,7]      # 每步积分后的action field
```

member轴始终保留，不先平均成一个Program。member reliability、stage progress和跨member disagreement分别保存，形成
particles或其等价的mean+structured covariance。Stage 1A只回答teacher等价类是否完整、非退化且覆盖真实闭环状态，不以
LoRA geometry作为通过条件。

本轮Stage 1A已通过：五个独立members逐task均有strict success，Goal/Long在内的五个48-state banks全部完整。

## 5. Stage 1B：occupancy-complete privileged realization oracle

该关键GPU实验已经完成。它不读取视频、不学习shared weights、不形成deployment route；held task各自使用privileged
effect particles，检验一套rank16 LoRA能否实现该策略等价类。

### 5.1 Stable carrier与无交叉项有效更新

首轮固定使用已验证的fit19 stable carrier（held5 strict250为43）并始终报告carrier-only。对每个LoRA target保留carrier
的rank16 `A_c`，只求一个task-local `Delta B`：

```text
W_task = (B_c + Delta B) A_c
       = B_c A_c + Delta B A_c
```

因此：

- `Delta B=0`精确恢复carrier behavior；
- correction在effective-update坐标中严格相加，没有`Delta B Delta A`或其它A/B交叉项；
- 输出天然仍是一套完整rank16 LoRA，不需要rank32 union、第二adapter或最终LoRA平均；
- fixed-A是首轮明确的capacity约束，不被包装成已证明的最终最优参数化。

### 5.2 Distributional objective

候选响应`R_theta(s)`不回归member均值。对每个state category与stage，先计算候选到每个member particle的归一化
owner/flow/action距离，再用固定temperature的soft minimum形成等价类损失。member assignment在同一stage内共享，避免每个
tensor元素任意拼接不同策略；不同stage允许选择不同成功member，表达阶段性policy composition。

总目标由四部分组成：

1. particle-equivalence：接近至少一个成功member的完整owner/flow/action response；
2. carrier no-worse barrier：相对同一particles的误差不得比carrier更差；
3. source/shared preservation与effective-update trust region；
4. category-balanced、stage-progress-aware weighting与member-disagreement uncertainty。

固定solver复用PECS已验证的12-step inverse-sqrt-decayed owner-normalized更新，但只优化`Delta B`。首个单task profile只允许
修正OOM、batching或明显量纲错误；profile不是科学结果。profile后solver step、temperature、category weights和state IDs全部
冻结，held task不early-stop、不挑step、不保留optimizer。

### 5.3 直接闭环Gate 2

第一次完整设计后直接在原held5 fixed250上评测final step12，不由geometry预筛。只有final12接近通过时，才补step10/11
作为相邻稳定性证据；final选择不变。

Stage 1B通过必须同时满足：

- absolute至少达到direct-earliest参考`74/250`，且相对carrier43净增至少20；
- 5/5 task非零，至少4/5 task严格高于carrier；
- Goal与Long各自非零；
- 相对多member success union的oracle-normalized recovery整体至少0.35，至少4/5 task为正；
- carrier success retention至少`33/43`；
- exact-row retention只作辅助，own-LoRA cosine/retrieval只作诊断；
- 配对字段、single-LoRA与信息墙合同成立。

如果final12明确失败，立即暂停分析，不自动创建successor、扫小超参或把profile写成科学版本。失败只说明当前
`state support + particles + fixed-A carrier + fixed solver`组合未建立强realization upper bound；由于当前仍缺真正
process-identifying meta tasks，它不会被夸大为整个EMBER目标的最终反证。

正式结果为`78/250`，逐global0/9/18/25/36为`36/12/30/0/0`。absolute、相对carrier净增与carrier retention通过；breadth、
4/5严格胜carrier、Goal/Long及oracle-normalized recovery失败，故按上述合同暂停。精确裁决见
`docs/evidence/ecp_20260823/ecp_occupancy_complete_oracle_gate_20260823.json`。

### 5.4 Fixed-A capacity已被闭环裁决

对latest、independent、earliest三个已知成功members逐target求解析最优fixed-A投影：

```text
B_star = B_expert A_expert A_carrier^T (A_carrier A_carrier^T)^+
W_projected = B_star A_carrier
```

三个strict250结果为`49/41/35`，相对matched direct只保留`31/108、22/113、14/74`，合计
`67/295=22.71%`。Goal原有24个direct successes、Long原有11个，三个投影member均为0；Long的effective-update energy
coverage反而是五项最高之一。因此当前fixed-A行空间不是一个可继续靠objective校准挽救的中性载体，而是行为上binding的
参数约束。它从Stage 1B主线退出，但这个结论不外推为“数学上不存在任意fixed-A功能等价解”。

下一realization operator必须同时满足：

- zero correction在effective-update层面精确返回stable carrier；
- correction显式加到effective update，不通过同时raw更新A/B引入交叉项；
- task-conditioned row与column space都可移动；
- 每步或最终用确定性rank16 retraction返回一套38-target LoRA，不部署rank32 carrier或第二adapter；
- 首轮保持同一48-state三particle objective与held5 gate，只分离parameterization；
- 在实现前说明其更新几何、数值profile与停止条件，不能把free A/B当作新一轮raw-factor小扫。

正式证据见`docs/evidence/ecp_20260823/ecp_fixed_a_capacity_gate_20260823.json`。

### 5.5 Mobile-rank4 residual先做容量门

资产审计进一步确认stable carrier不是有效rank16，而是精确rank12：38/38 targets的后4个B columns均为0，同时A的16行
full-row-rank。因此最小的row-space-mobile successor不是立即自由更新整套A/B，而是保留前12个carrier ranks，并用后4 ranks
表达任意行、列空间的effective correction：

```text
W_task = B_c12 A_c12 + B_r4(task) A_r4(task)
```

这与历史phase-code residual4的参数拓扑相同，但科学问题不同：历史`37/33`关闭的是当时learned code与functional decoder；
现在先把三个已知成功members的`W_expert-W_carrier`逐target做best-rank4 SVD。15个member-task的离线correction energy coverage
为`99.49%--99.69%`、expert update coverage为`95.34%--98.90%`，明显优于fixed-A，但仍必须直接闭环。

当前唯一授权动作是三个预固定members各自的strict250解析容量评测。通过后才为现有48-state objective写rank4 residual solver；
失败后才讨论full-rank16 retraction。精确合同见`docs/ecp_rank4_residual_capacity_card_20260823.md`。

正式结果为`110/120/76`，逐arm均略高于matched direct `108/113/74`且5/5 tasks非零；pooled matched retention为
`245/295=83.05%`，Goal为`15/24=62.5%`。但Long虽由absolute合计11提高到20，matched-member exact-row retention只有
`4/11=36.36%`，故没有通过预注册50%条款；capacity-binding条款也全部未触发，裁决为mixed。三个member集合层的Long
direct/projected success union为`11/16`、overlap6、union retention`54.55%`，说明差异集中在policy/member identity，而不是
Long能力为0。该post-hoc定位符合专家“exact row只作辅助”的提醒，但不重写预注册门，当前仍暂停solver授权。正式证据见
`docs/evidence/ecp_20260823/ecp_mobile_rank4_residual_capacity_gate_20260823.json`。

## 6. Stage 1C：被当前non-pass阻止

只有未来新的强realization oracle通过后，才允许学习：

```text
ECP Program
    -> event/layer/family policy-effect distribution
    -> frozen shared realization operator
    -> Delta B
    -> one complete rank16 LoRA
```

privileged occupancy在训练中监督共享effect basis、uncertainty和realization operator，但在deployment时被边缘化；部署前向
仍只读取`P_lang/P_scene/P_process/rho/sigma`。首个共享实验前必须：

- 轮换train24 fold，避免继续用已高度审阅的fold0选择shared模型；
- 加入经审计的source-unseen adaptation meta tasks；
- 若要声称process understanding，至少一部分数据必须真正满足same endpoint/different required procedure；
- 先校准任何重复使用的open-loop proxy与closed-loop相关性。

Stage 1C通过后才进入Dynamic-K `q_V`、除backbone外普通Writer参数的联合训练、structured outer credit和validation8
single-checkpoint development evaluation。联合训练不是遗漏的永久禁区，但不能在policy realization坐标尚未成立时再次让
Program与compiler共同旋转。

## 7. 生命周期与停止条件

active tree只保留一条ECP路径。每个重大科学问题对应一张falsification card；bugfix不增加版本，profile不算科学裁决，
每个主要家族第一次完整实现后必须直接closed loop。

只有在一次真正强的oracle同时具备process-identifying source-unseen data、多个独立成功策略、initial/successful/candidate/
recovery occupancy、effect distribution、stable carrier和fixed realization后仍不高于shared、breadth不超过3/5、Goal/Long为0、
且只恢复很少task-local oracle gains，才停止广义`action-hidden video -> static full LoRA`主路线。届时优先转向runtime
video-conditioned policy、video-to-reward/progress、video-initialized task-local RL或skill/subgoal composition。
