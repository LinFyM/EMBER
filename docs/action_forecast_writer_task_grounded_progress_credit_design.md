# Task-Grounded Semantic Progress Credit Writer 设计

状态：**2026-08-05只读与Writer-update profile门均通过；正式run已封存为fresh AS125、首段stop1。**

本设计接续`docs/action_forecast_writer_relative_flow_credit_design.md`的binary-only负裁决。
它不把RL当作绕过LoRA质量问题的替代路线，而是专门修复已经定位到的最早接口：
teacher-video条件虽然能生成幅度足够、彼此不同的LoRA，但Writer没有获得“哪一种条件变化
在真实closed loop里有用”的内容级信用。新方法仍是one-shot Writer；只在train24的
official random-reset rollout中，用action-hidden teacher video、task language和机器人
实际看到的第三人称RGB建立相对信用。

## 1. 已封存的起点与负裁决

唯一cold start是同一fresh v6 AS root的macro125：

```text
runs/outputs/pi05_as_writer_v6_relative_flow_coldstart_formal_r6_b20_seed7_b75cb19_20260804/checkpoints/step_00000125
```

它完成125个full24 macros、60,000 queries和3,000 one-video conditions；source policy、
normalization和validation/test information wall完整。step25/50/75/100/125的K4
success为`25/38/47/52/50`，coverage为`12/14/18/17/19`。task36/38/39在五点始终
`0/4`，step125仍有5个all-failure tasks，因此只有binary LOO advantage的Writer-RL
无法给这些任务任何方向。

step100→125内部审计进一步排除了“失败只是LoRA太小或视频没被读到”：LoRA norm中位
`99.18→109.11`，但same-task video energy`.1300%→.1154%`、demo间BA差异
`.0475→.0448`、fixed-action demo差异`.0101→.0086`都没有增强；持续全失败组的视频
差异反而更大。成功数变化与video-energy变化Spearman为`-.521`。所以必须改的是
condition-to-policy credit，不是继续放大norm、rank、store数量或随机条件差异。

binary profile的ratio、PPO/SPO、failure replay、六卡deferred NCCL和exact-resume均已
实证健康。它们作为工程底座保留；任何profile cycle checkpoint都不得作为本设计初态。

## 2. 核心假设

本设计只检验一个新假设：

> v6 AS125已经学到可用的task-grounded视觉变化，但binary success对all-failure任务
> 太稀疏。若用冻结、policy-aware的视觉语义前端，把rollout从自身起点发生的内容变化
> 投影到同task teacher video的内容变化方向，就能在不读teacher action、pose或视频
> 时钟的前提下，为all-failure轨迹提供有符号相对信用；下游Writer再把这种信用映射到
> closed-loop有效的LoRA方向。

该假设失败时，不能把结果解释成“RL不适用”；它只拒绝当前冻结语义表征能够充当
progress observer。届时应重做action-free value representation或condition-to-policy
接口，而不是降低门、加LIBERO reward shaping或恢复监督functional loss选择。

## 3. 三个角色必须分离

### 3.1 Source policy

source policy、normalization和public rank-16 topology继续完全冻结。它只执行Writer
生成的一套LoRA，并为现有CFM old/current ratio提供functional evaluation；其loss数值
不选择checkpoint。

### 3.2 Writer

Writer部署输入仍严格是：

```text
pure task language + exactly one action-hidden same-task teacher video
```

不新增rollout observation、reward、terminal、task ID或第二条视频作为Writer输入。
AS125之后永久关闭teacher action入口。若只读门通过，RL阶段只训练
`semantic_core`、`visual_transition`、`procedure`、`compiler`和`factor_heads`；
`semantic_encoder`永久冻结。

### 3.3 Frozen progress observer

progress observer直接复用AS125后冻结的同一个`semantic_encoder`，不复制第二套可训练
视觉模型，也不随RL更新。它只读：

- 纯task language；
- 当前one-shot teacher video的旋转后agentview RGB；
- 同task official rollout的旋转后agentview RGB。

它禁止读取teacher/rollout action、proprio、reward、terminal、object pose、文件名、
task ID、init-state ID、policy hidden state或validation/test数据。official binary success
只在observer输出之后决定哪种advantage拥有优先级，不能成为observer输入。冻结前端既
阻止reward hacking，也把问题限定为“如何把已有语义条件写进policy”。

## 4. Canonical content potential

只读诊断与首个可能的Writer profile只允许下面这一套potential，不并行扫描多个reward。

对task language `l`与一帧第三人称RGB `x`，冻结semantic encoder输出：

- 每个有效task token的task-queried patch evidence `g_k(x,l)`；
- 固定Action-Expert noise probe产生的interaction evidence `h(x,l)`。

不使用包含静态language residual的完整frame evidence。将每个token向量与interaction向量
分别做RMS normalization，记为`z_c(x,l)`；component集合是所有有效task tokens加一个
interaction component。teacher视频只取真实第一帧`t0`和真实最后一帧`tG`，定义：

```text
d_c = z_c(tG,l) - z_c(t0,l)
w_c = ||d_c||^2 / (sum_j ||d_j||^2 + eps)
```

对一条rollout自己的第一帧`r0`与任一当前帧`r`：

```text
x_c(r) = z_c(r,l) - z_c(r0,l)
q_c(r) = cosine(x_c(r), d_c) * min(||x_c(r)|| / (||d_c|| + eps), 1)
Phi(r | l, teacher) = sum_c w_c * q_c(r)
```

零向量component贡献0；`Phi`有界于`[-1,1]`。canonical episode utility是terminal frame
的`U=Phi(r_terminal)`，不取trajectory maximum，不用episode length、frame ordinal、
normalized video time或environment reward塑形。start-relative delta用于抵消teacher与
official random reset之间的初始背景和物体姿态差异；teacher-change energy自动降低未变
token的权重。reversed会交换`t0/tG`并翻转内容方向，shuffled会真实改变首尾内容，wrong
video会改变teacher方向，因此三种反事实仍能检验视频因果性。

endpoint potential是有意的首个最小机制：它使用task language、teacher展示的真实状态
变化和policy-aware interaction，却没有人工标注“第几步应该完成什么”。若它未过门，
不能在同一结果上临时加入milestone、DTW、视频时间回归或task-specific子目标救分；下一
方法必须fresh封存。

## 5. Binary-first task-relative advantage

每个task/teacher condition仍做K4 official random-reset trajectories。credit优先级固定：

1. **mixed binary**：至少一成一败时，完全沿用official binary K4 leave-one-out
   advantage；semantic utility只作诊断，不改变成功高于失败的顺序。
2. **all-success**：严格零梯度，避免dense observer破坏已经稳定成功的条件。
3. **all-failure**：使用`U_i`的leave-one-out差：

```text
A_i = U_i - mean_{j != i}(U_j)
```

`U`本身有界，首个profile不做rank normalization、temperature、hand-tuned scale或
confidence gate。若同task四条utility相同，则自然零梯度。正advantage继续用per-CFM-
sample PPO clip，负advantage继续用SPO pullback；每episode先等权、再task等权、最后
full24 sum。

同一LoRA在整条episode中固定，因此首个实验把episode advantage等权赋给该episode的
全部executed action-prefix ratio samples。这是在condition-level评价一套LoRA，不需要
引入另一个state-dependent critic或手工chunk reward-to-go。以后若Writer变为
state-dependent，必须重做本合同，不能沿用这一推导。

## 6. 只读机制诊断（先于任何梯度）

诊断固定使用AS125、现有step125 K4的相同train24 task/video/env/policy seeds与K4预算。
它可以重放确定性rollout以捕获第一帧和terminal agentview，但必须在optimizer、Writer
backward和checkpoint创建前结束。每条row保留binary outcome、`U`、各component energy、
correct/wrong/shuffled/reversed counterfactual utility和必要的pixel-change nuisance；
不保留proprio、action以外的新敏感payload，也不读取target actions。

以下门在看到新utility前预注册，全部通过才允许Writer-update profile：

0. **paired K4身份门**：同一AS125确定性重放必须保持50/96 successes、14 mixed、
   5 all-success和5 all-failure；否则先裁决environment/policy pairing，不能解释utility。
1. **finite/content门**：24/24 correct teacher directions finite，且每task总teacher
   change energy严格大于`1e-6`；重复同输入forward的utility绝对差不超过`1e-5`。
2. **binary agreement门**：step125的mixed tasks中，至少`10/14`满足success mean
   utility严格高于failure mean；汇总所有同task success-failure pairs的AUC至少`.60`。
3. **all-failure dispersion门**：5个step125 all-failure tasks中至少4个utility range
   达`.05`，五task range中位至少`.10`；长期全失败task36/38/39中至少2个达`.05`。
4. **video counterfactual门**：在step125 successful trajectories上，correct utility
   分别高于wrong、shuffled、reversed的比例都至少`.65`，且三种paired margin中位都
   严格为正。
5. **非像素捷径门**：20条all-failure trajectories中，utility与raw terminal-start
   pixel L2的Spearman绝对值低于`.80`。observer contract另行确认没有step、length、
   frame index或proprio tensor进入forward。

上述门使用train reward仅验证observer是否与已有official outcome相容；不读取或选择
validation/test结果。任何一项失败都封存为机制负结果，不启动Writer梯度，也不按单task
调阈值。

## 7. Writer-update profile门

只读门通过后才允许从AS125权重复制到全新、不兼容的profile root，做一个full24 cycle、
K4、Nmc4和两learning epochs。profile必须同时证明：

- observer/`semantic_encoder`、source policy和normalization trainable/grad均为0；
- mixed tasks的binary advantages逐项与旧owner相同；all-success为0；all-failure只来自
  上述semantic LOO；
- 至少4个all-failure tasks产生finite nonzero gradient；五个Writer下游主block可达；
- ratio、clip、grad norm、OOM、NCCL readiness与完整cycle checkpoint健康；
- `NCCL_P2P_DISABLE=1`、实际world size、rank ownership与sealed 3+3 NUMA topology不变；
- peak reserved适配约46GB A40，不通过减少K/Nmc/scientific batch掩盖OOM。

profile checkpoint仍不得续训。只有profile完成后，才根据梯度覆盖、wall与显存封存formal
cycle数和评测点；正式run必须从AS125重新fresh进入本schema，而不是从profile或旧binary
cycle1 warm-start。

## 8. Formal结果如何裁决

正式checkpoint选择仍只认single-checkpoint paired closed-loop correct400、breadth和
能力换手。训练success、semantic utility、functional ratio loss和漂亮LoRA几何都不能
单独选点。

最低成功标准不变：同一single checkpoint的paired correct严格`>150/400`，达到后继续
追求更高absolute、breadth和视频因果性。winner必须补correct/same/wrong/shuffled/
reversed；如果wrong/shuffled/reversed同步上升，说明credit学到task-static或视频无关
方向，不能称为EMBER改善。

若formal train reward改善而held correct下降，裁决为train24 occupancy overfit；若只有
all-failure训练task改善而原mixed/all-success能力大量丢失，裁决为semantic tie-break
破坏共享Writer；若utility门通过但Writer梯度仍只形成大而off-manifold的LoRA，最早失败
接口回到downstream compiler/factor到source-policy tangent的对应，而不是observer。

## 9. 为什么这不是监督微调特化trick

- observer只依赖action-free video、language和online visual observation；监督学习、
  policy gradient、actor-critic或其它相对policy objective都可复用同一potential。
- binary-first规则是对真实任务目标的优先级，不依赖functional action label或SFT loss。
- 冻结observer把表示学习和policy improvement解耦，避免奖励随Writer一起漂移。
- start-relative、task-token grounding和policy-aware interaction是输入/表示设计，不是
  LIBERO object pose、人工子目标或某个task的reward shaping。
- Writer部署时仍是one-shot前向，不需要环境交互、critic或额外视频。

因此本设计是“action-hidden demonstration定义任务变化，closed loop决定哪种Writer写出
有效”的通用接口。LIBERO只提供当前评测环境和official binary最高优先级，不进入
potential的对象名称或状态机。

## 10. 与现有action-free video reward工作的关系

VIP从无action视频学习value-implicit representation，并将冻结表示用于goal-image dense
reward；LIV把language-image对齐加入action-free value representation；RoboCLIP展示了
单条视频或文本演示经冻结视觉语言表示生成RL reward的可行性：

- VIP: <https://arxiv.org/abs/2210.00030>
- LIV: <https://arxiv.org/abs/2306.00958>
- RoboCLIP: <https://arxiv.org/abs/2310.07899>

这些工作只支持“action-free demonstration可提供视觉信用”这一可行性，不证明PI05
AS125 hidden state天然是value representation。EMBER当前差异在于：同一个one-shot
teacher video既生成整套policy LoRA，又在训练期定义task-grounded、start-relative的
condition-level credit；最终仍由single-checkpoint multi-task closed-loop性能裁决。
novelty只有在机制门和正式结果成立后才能主张。

## 11. Schema、resume与artifact边界

- canonical RL config后续原位升为fresh incompatible schema；旧binary config/owner只由
  Git、本设计前的artifacts和relative-flow design保存，不保留第二个active launcher。
- run contract必须封存AS125 checkpoint identity、冻结frontend参数集合、potential公式、
  eps、K/Nmc、counterfactual mapping、actual world size、physical/NUMA mapping和GPU
  transport。
- checkpoint保存Writer下游参数、optimizer/scheduler、task/video/env/policy/flow RNG、
  complete-cycle cursors、全rank ownership与rollout/utility ledger cursor；observer不
  单独训练，但其AS125 provenance必须可重建。
- 只读diagnostic、profile和formal root分离；不得把前两者冒充正式结果或互相resume。
- 不做重复全量hash。只在新schema authority、formal checkpoint seal或真实identity冲突
  时核验对应identity；其余使用既有manifest和run contract。

## 12. 当前执行顺序

1. seal本design与活动文档；
2. 原位实现只读observer、endpoint frame retention和utility/counterfactual ledger；
3. 跑聚焦CPU合同与最小GPU shape/memory vertical path；
4. live比较`gpu01/gpu02`，只用空闲卡且总数不超过6，重放AS125 K4只读机制诊断；
5. 若任一预注册门失败，封存负结果并重新分析，不启动Writer update；
6. 全部门通过后才实现/运行一个全新Writer-update profile；
7. profile再决定formal训练预算和paired correct400评测。

## 13. Formal首次启动暴露的collective入场根因

首次AS125-fresh formal0→1完整产生96 rollout与24 task progress-credit，但第一轮
gradient sum中rank0/1/2/5进入NCCL seq18，rank3/4停在seq17，600秒watchdog终止；没有
optimizer update、metrics或checkpoint。task ownership、outcome分组与profile完全一致，
排除科学数据变化；旧`FileStore` barrier只覆盖Python enqueue且其临时文件生命周期在
高度错峰时不能可靠证明所有rank的CUDA工作已结束。

canonical工程合同因此升级为：每rank先显式CUDA synchronize，再按本次torchrun唯一
session、cycle、epoch和rank写原子marker；只有实际world-size全部marker可见后才进入
NCCL。marker在run内保留，新launch使用新session隔离旧状态。相同输出目录连续两个真实
六卡session已分别完成6/6 marker和all-reduce sum21。该修改不改变rollout、semantic/
binary credit、K4/Nmc4、task等权或optimizer；正式接受仍要求全新root重放原96-rollout、
two-epoch规模并产生finite update、双ledger checkpoint和exact-resume证据。

clean/pushed`30977b5`已完成该原规模重放：epoch0/1分别形成6/6 marker后进入NCCL，
2次finite update、完整cycle1 checkpoint、0 watchdog/OOM，六rank rollout/progress-credit
双ledger通过validator。该证据接受新的collective入场合同；它仍不证明Writer性能提高，
因此下一裁决保持为AS125 baseline与cycle1 strict paired correct400，结果前不续cycle2。

## 13. 只读机制裁决与profile授权

clean`c483497`上的六卡只读root为
`runs/outputs/pi05_task_grounded_progress_credit_diagnostic_as125_r6_c483497_20260805`。
它严格重放AS125的96条K4身份，得到50 successes、14 mixed、5 all-success与5
all-failure；旧profile的task/cursor、env/policy seed、初态、demo、LoRA及outcome逐项
一致。运行写入0 optimizer update、0 Writer backward与0 checkpoint，wall
`401.874s`，峰值reserved`19,289,604,096` bytes。

全部预注册门通过：mixed success均值高于failure为`13/14`，同task pair AUC=`.8913`；
task4/20/36/38/39的utility range分别为`.1228/.5712/.3338/.2554/.2371`；successful
rollout上correct优于wrong/shuffled/reversed比例=`1/.88/1`，margin中位
=`.4889/.3557/1.6208`；all-failure utility与raw pixel-change Spearman=`.5564`。
此外20条all-failure rollout上correct优于三反事实比例均为1，说明信号不只由已有成功
样本支撑。

该结果只接受“冻结AS125语义前端可提供有内容、task/video特异且非纯pixel的相对
progress credit”，不接受“Writer LoRA已经改善”。现在只授权一个从AS125 fresh进入的
full24、K4、Nmc4、two-epoch profile；profile checkpoint禁止续训。profile通过冻结、
梯度覆盖、ratio、NCCL与A40显存门后，才新增formal seal与训练预算。

## 14. Writer-update profile裁决与formal seal

clean`84d856c`的fresh profile root为
`runs/outputs/pi05_task_grounded_progress_credit_writer_profile_as125_r6_84d856c_20260805`。
它完成96 rollout、24,593 actions、50 successes、14 mixed、5 all-success与5
all-failure；两epoch wall`2129.187s`，peak reserved`19,455,279,104` bytes，0 OOM、
clip、watchdog与observer gradient。19个active credit tasks严格由14 mixed binary加5
all-failure semantic组成，五个all-failure task在两epoch均有finite nonzero
generated-LoRA gradient。

五个Writer下游block在两epoch均可达。epoch0/1梯度范数分别为：semantic core
`.00218/.00350`、visual transition`2.76e-5/4.33e-5`、procedure`.000240/.000429`、
compiler`.00457/.00734`、factor heads`.03680/.05461`；总grad norm`.03715/.05521`。
ratio范围=`[.99077,1.02504]`与`[.74545,1.09294]`、mean近1且positive clip均0。
两轮都按真实负载不均完成FileStore ready再NCCL sum。

相对只读诊断，95/96 rollout的完整steps/noise prefix一致；唯一task28/cursor1保持相同
初态、LoRA、seed和成功outcome，但在第16而非17个replan chunk终止，少7 actions。五个
all-failure task的20条utility相对诊断保持全部task内排序，最大/平均绝对差
=`.01622/.00318`。该差异裁决为成功边界的环境终止时刻微扰，不改变outcome group、
credit owner或机制门。

formal固定从AS125重新fresh，6 ranks、K4、Nmc4、two epochs，总上限8 cycles，checkpoint
=`1/2/4/8`。首段只跑0→1；随后在同一strict panel上比较AS125 baseline与formal cycle1
的paired correct400、breadth和task gained/lost，再决定是否exact-resume到2/4/8。
profile checkpoint永久禁止续训或评测。formal checkpoint另绑定每rank rollout与
progress-credit双ledger前缀；任何续段不得改变两epoch、task/video schedule或拓扑。
