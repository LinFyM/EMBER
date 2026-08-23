# EMBER-ECP: Event-Conditioned Policy Compiler

状态：2026-08-24 **正在接受专家全过程复核；新的代码/GPU推进暂停。既有direct-effect realization已裁决，但此前把
effect bank误写成完整Stage 1A pass、把绕过Program的solver误写成完整Stage 1B，现已校正。**
旧ECP Stage 1 v1--v24、MDCO、PECS及本轮oracle的精确实现、结果和退出原因由`docs/research_history.md`、Git与formal
artifacts保存；最新裁决为
`docs/evidence/ecp_20260823/ecp_mobile_rank4_solver_gate_20260823.json`。
本次方案对齐与待专家裁决问题见`docs/ecp_expert_alignment_audit_20260824.md`。

## 1. 当前裁决

ECP的核心目标与Stage 0候选表示尚未被证伪，但以下**实际检验过的实现组合**已经有足够反证，不再恢复或做小变体：

- 历史deterministic/mean `q_pi`；
- learned Program-to-A/B hyperdecoder；
- 19或90个mapping上的21M target/rank query compiler；
- v24、MDCO以及width/rank/head/fusion/LR/seed后继；
- 只在教学视频稀疏support frames上拟合exact policy effects的PECS；
- 当前48-state三particle effect bank、stable fixed-A carrier与12-step Delta-B realization的组合；
- 当前从exact-zero residual开始、直接更新mobile-rank4 raw A/B factors的12-step realization operator；
- 用open-loop geometry、own retrieval或LoRA cosine替代早期closed loop。

本轮direct-effect privileged子问题已经得到否定答案：

> 在真实闭环状态分布上，用多个独立成功策略定义一个分布式策略等价类后，固定、受约束的rank16实现器能否从稳定
> carrier出发，恢复显著且广泛的task-local闭环能力？

fixed-A结果由carrier `43/250`提高到`78/250`，但breadth仅3/5、Goal/Long为0、oracle recovery `.304`，整体Realization
non-pass。解析容量分离随后证明fixed-A行为上binding，而mobile-rank4 topology能以三个known-success members达到
`110/120/76`。但保持同一bank/objective/12-step数值的mobile raw-factor solver只有`49/250`，逐task`40/3/6/0/0`，
member-union gap只恢复`3/115`。它停止在接近carrier的极小修正，并未进入known-success effect basin。因此当前不训练video
predictor、shared compiler或Writer联合模型，也不扫solver、rank、step、LR、初始化或权重。

这项结论不能外推为完整privileged ECP链失败：当前retained Stage 1直接读取effect bank，没有distributional `q_pi(P)`，也没有
`Program -> event/layer/family policy-effect distribution`模块。gauge-invariant realization、effect objective充分性与缺失的
Program bridge必须由专家重新确定依赖顺序。

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

retained Stage 0另输出language summary与scene transition，但当前没有把它们固定为贯穿后续链路的
`P_lang[38,128] / P_scene[38,128]`同构Program。这个缺口不能由effect bank或solver自行补齐。

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

## 4. Stage 1A evidence与尚未实现的distributional policy teacher

Stage 1A不生成LoRA。已完成部分建立了有显式member轴、state-support轴和stage轴的privileged evidence bank；专家最终方案
还要求由这些evidence产生与`q_V`同构的distributional `q_pi(P)`，这一后半部分尚未实现。

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

本轮只通过Stage 1A的evidence prerequisite：fold0五项各有一个新seed37独立member并取得strict success，配合既有两类
members形成每task三particle；Goal/Long在内的五个48-state banks全部完整。它不能再被简称为“Stage 1A整体通过”。

### 4.4 尚缺的`q_pi(P)`

专家要求的privileged teacher应读取multiple successful adapters、successful/learner/recovery occupancy、policy responses、
stage与uncertainty，输出与deployment video encoder完全同构的posterior：

```text
q_pi(P) = distribution over {
  P_lang[38,128],
  P_scene[38,128],
  P_process[8,38,128],
  rho[8],
  sigma[8,38,128]
}
```

posterior应保留mixture/particles或mean+structured covariance，并区分video-observable与rollout-only recovery information。
当前`Stage1EffectBank`保存的是prefix/noise/category/stage/progress以及source/carrier/member owner/flow/action effects；仓库没有
实现上述Program posterior的retained `q_pi`模块。

## 5. Direct-effect realization子门：occupancy-complete privileged oracle

该关键GPU实验已经完成。它不读取视频、不学习shared weights、不形成deployment route；held task各自直接使用privileged
effect particles，检验一套rank16 LoRA能否实现该策略等价类。`stage1_oracle.py`加载`effect_bank_manifest`后直接进入solver，
没有Program或Program-to-effect forward，因此这里只是lower-level realization子门，不是完整Stage 1B compiler gate。

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

### 5.3 Direct-effect closed-loop gate

第一次完整设计后直接在原held5 fixed250上评测final step12，不由geometry预筛。只有final12接近通过时，才补step10/11
作为相邻稳定性证据；final选择不变。

该direct-effect子门通过必须同时满足：

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
参数约束。它从direct-effect realization候选中退出，但这个结论不外推为“数学上不存在任意fixed-A功能等价解”。

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

### 5.6 Rank-reserved mobile-rank4 realization

原mixed verdict不重写。专家要求的selection unit是successful-policy equivalence class，而非某一个member的exact row identity；
本轮absolute、5/5 breadth、Goal/Long非零与multiple-member union已经排除明显mobile-rank4 topology binding。因此下一唯一
问题是：同一48-state three-particle effect objective能否通过固定12-step solver找到该参数面中的闭环有效区域。

operator固定为`W=B_c12 A_c12 + B_r4 A_r4`。carrier12冻结，`A_r4`从carrier未激活tail rows初始化、`B_r4=0`；A/B按target
联合归一化更新并在每步后只对residual做balanced compact-SVD regauge。step0精确为carrier，final仍是一套rank16。完整单变量
合同与原Gate见`docs/ecp_mobile_rank4_solver_card_20260823.md`。

该实验已完成并明确non-pass。五项inner objective均严格下降，但strict250只有`49`，逐task`40/3/6/0/0`；相对carrier
retained/gained/lost为`41/8/2`，union recovery只有`3/115=.0261`。known-success member effects在同一objective上有
`.060--.163`低值，known-success mobile projections的trust为`1.341--2.281`，而solver final trust仅
`.000915--.001171`，correction方向cosine仅`.041--.077`。所以这轮关闭的是zero-start raw-factor dynamics，不是mobile-rank4
容量，也尚未关闭全部effect-distribution objective。当前不补相邻step、不调数值、不训练video predictor或Writer。

### 5.7 Effective-update reachability oracle

下一单变量实验不放大raw-factor LR，也不增加VJP预算。它在exact carrier处用固定8列input sketch和4次matrix-free VJP近似
dense effective-update gradient的rank4最陡下降方向；首步后最多8次VJP使用Gram-preconditioned tangent
`dB A + B dA`，每次只在低秩core中retract为rank4。全部候选用同一objective和固定
`1,1/2,1/4,1/8,1/16` trust backtracking单调接受，mean trust继续不超过1.5，总VJP最多12。

非held ordinal71必须先恢复至少50%的carrier-to-best-member objective gap，且trust进入`.10--1.50`；否则在held rollout前
停止。profile通过后仍直接使用原strict paired250 Gate，不用inner objective选择LoRA。精确合同见
`docs/ecp_effective_update_solver_card_20260823.md`。该卡仍不授权Program、Writer或Action Meta；realization成立后再按owner要求
做一次matched Action Meta control，无负面才启用并冻结。

正式profile从exact carrier得到finite负方向导数`-81.8873`，但五个固定回溯尺度都没有使完整objective严格下降，故0步接受、
gap recovery与trust均为0，只消耗4次initial sketch VJP。按预注册门，这个operator在进入Gram tangent与held5前停止。该结果只
关闭当前matrix-free sketch归一化、固定回溯网格与effect objective组成的solver，不关闭mobile-rank4解析容量或ECP核心；也不能
用“方向导数为负”事后缩小步长继续搜索。证据见
`docs/evidence/ecp_20260823/ecp_effective_update_profile_gate_20260823.json`。

## 6. 完整Stage 1与`q_V`：等待专家重新定稿

当前不是“Stage 1C只被一个solver non-pass阻止”，而是完整privileged链还有两个未实现接口：

```text
successful-policy evidence
    -> distributional q_pi(P)
    -> event/layer/family policy-effect distribution
    -> fixed shared realization operator
    -> one complete rank16 LoRA
```

下一次held5 gate必须冻结哪些模块、direct-effect oracle在其中是什么前置角色、realizer是否应先独立通过，均由本轮专家复核
重新裁决。仓库在得到这个顺序前不创建successor。

只有完整privileged链显著高于shared、覆盖Goal/Long并保留multi-member support，才进入同构deployment posterior：

```text
q_V(P | exact language, action-hidden ordered videos)
    -> frozen Program-to-effect compiler/realizer
    -> one complete rank16 LoRA
```

之后才允许普通Writer参数联合训练、structured outer credit与validation8 single-checkpoint development evaluation。联合训练不是
永久禁区，但不能再次同时发明video semantics、Program gauge和realizer坐标。

### 6.1 Process-identifying data是资格缺口，不是已确定的唯一下一主线

现有全部授权BDDL只按最终状态合取判success，无法让“同language/endpoint、不同必需process”成为训练标签。若要声称general
process understanding，仍需经审计的source-unseen paired variants；可行性见
`docs/ecp_process_identifying_meta_task_feasibility_20260823.md`。

但当前证据只说明这类数据缺失，不证明它是所有失败的唯一根因。它应在distributional `q_pi(P)`之前建立，还是只在`q_V`与
最终process qualification前加入，是本次请专家明确的问题；未定稿前不创建custom tasks或启动GPU。

## 7. 生命周期与停止条件

active tree只保留一条ECP路径。每个重大科学问题对应一张falsification card；bugfix不增加版本，profile不算科学裁决，
每个主要家族第一次完整实现后必须直接closed loop。

只有在一次真正强的oracle同时具备process-identifying source-unseen data、多个独立成功策略、initial/successful/candidate/
recovery occupancy、effect distribution、stable carrier和fixed realization后仍不高于shared、breadth不超过3/5、Goal/Long为0、
且只恢复很少task-local oracle gains，才停止广义`action-hidden video -> static full LoRA`主路线。届时优先转向runtime
video-conditioned policy、video-to-reward/progress、video-initialized task-local RL或skill/subgoal composition。
