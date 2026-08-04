# Task-Relative Flow-Credit Writer

状态：2026-08-04 BCI design authority。本文在 Policy-Target-Owned Factor 的
`99/76/86/68`负裁决及历史 RL-Writer 机制复核后建立。首个实现必须恢复已经封存、
单 checkpoint 上限最高且已证明能传播时序信息的 v6 Writer，随后以一条新的
task-relative on-policy flow-credit 路径原位替换退役的 success-filtered RL-Writer；
不能并存第二套 Writer、reward runner 或 evaluator。

## 1. 决策

下一轮不再修改 LoRA head 数、层 scale、rank、谱或手工 SFT profile。当前最早失败
接口是 condition 到 policy 的训练 credit：同 task+demo 的 AS factor-gradient 方向
隔50 macro中位余弦只有`.0046`，而学到的异质 LoRA 主要落在 policy 不敏感方向。

三类候选中选择 task-relative flow credit：

1. 继续改 AS functional estimator：计算便宜，但既往 Latin/antithetic VR 只给很小的
   三步机制改善，无法给出 closed-loop occupancy credit；不选。
2. 直接在百万维 LoRA 或四千万维 Writer 上做黑盒 ES：reward 对齐，但 binary success
   下方向方差随维度过高；不选首轮。
3. 用 on-policy actions 的 conditional flow-matching ratio 形成 policy-gradient
   surrogate，并在同 task 内用成功与失败构造相对 advantage：直接使用目标闭环信号，
   同时沿 frozen source policy 的真实 action distribution 向 Writer 反传；选择。

该选择借鉴 Flow Policy Optimization 的基本关系和 FPO++ 的 per-sample ratio / asymmetric
trust region，但训练对象是生成完整 LoRA 的 Writer，而不是解冻 source policy：

- <https://arxiv.org/abs/2507.21053>
- <https://arxiv.org/abs/2602.02481>

## 2. 它不是历史 RL-Writer

历史 PI05 RL-Writer 的算法是：

```text
rollout -> 只保留成功episode -> 对自身成功executed prefix做普通flow regression
```

它没有失败 trajectory、advantage、old/current flow ratio或trust region。development
correct/wrong=`94/87`，视频因果弱；更早 SmolVLA self-imitation 还使 held success
`63→56→36→15/500`。所以不能把旧路径重新命名为新方法。

新算法是：

```text
同一task + 同一one-shot video + K条独立official random-reset rollouts
-> 成功和失败全部保留
-> task内leave-one-out binary advantage
-> old/current per-CFM-sample ratio
-> positive PPO / negative SPO trust region
-> 24 tasks等权、每epoch一次Writer update
```

失败 action prefixes 提供明确负 credit；容易 task 不会因更高 raw success rate 在全局
梯度中获得更大权重。reward phase 不读取 teacher action，也不训练 frozen policy。

## 3. Writer 架构与 phase 边界

### 3.1 恢复 v6，而不是继续 Target-Owned

恢复 frozen v6 的唯一 canonical 结构：

- task-grounded Semantic Core；
- Action Expert + adjacent visual-transition Causal Procedure；
- v6 slot fusion与完整rank-16、38-target public LoRA；
- functionally identity初始化、frozen source policy和同一normalization。

理由不是回滚历史结论，而是控制变量：v6-fast macro400 的 single checkpoint
correct=`143/400`，五臂=`143/135/125/128/129`；旧recipe又证明同一结构能形成强
order margin。它是当前最强且表示路径已验证的 reward-credit substrate。
Target-Owned ownership 虽成功解除跨层硬同向，但absolute只到99，因此不让一个已被
拒绝的capacity变量混入首个reward-credit实验。

恢复必须移植BCI已经根修的实际world-size task assignment、六卡collective、A40
microbatch和NCCL SHM合同；不得恢复A100四/八卡硬编码或旧绝对路径。

### 3.2 独立短 AS cold start

从 fresh identity v6 开始，按 fast-decay400 task-complete B20 做独立 cold start；
不加载历史 macro400、完整 AS best、Target-Owned checkpoint或`.pruned_init`。

每25 macro保存。只在24 train tasks做 official random-reset coverage probe，累计记录
每task first success、teacher-action queries、environment actions和wall。只有当前候选
在一轮 sealed exit probe 中24 tasks均至少一次success，才冻结 cold-start checkpoint；
随后永久关闭action data入口。coverage probe不读取validation/test reward。

若到macro400仍无法通过24-task exit coverage，本reward方案记为 cold-start coverage
不成立，不用完整AS best绕过。

### 3.3 Pure reward phase

reward phase的每个outer cycle固定覆盖全部24 train tasks。实际world size只决定每rank
持有几个task，不改变task集合或task权重。每个task在一个cycle中：

1. 无放回选择一条teacher video并生成一套LoRA；
2. 用同一LoRA做`K=4`条独立official random-reset rollouts；
3. 四条rollout使用不同environment/policy-noise cursor；
4. success与failure都保留实际执行过的action prefixes；
5. 未执行的chunk后缀永不获得credit。

## 4. Task-relative advantage

对同task/video group的binary return `R_i ∈ {0,1}`，使用leave-one-out baseline：

```text
b_i = mean_{j != i}(R_j)
A_i = R_i - b_i
```

该baseline不读取第`i`条trajectory自己的outcome，且同task内总和为零。若四条全成功
或全失败，本task本cycle梯度严格为零；不通过全局baseline让易task压过难task，也不在
没有相对行为证据时制造更新。

同一episode的所有replan chunks共享`A_i`，先在episode内部对实际executed chunks等权
平均，再对有非零advantage的episodes平均，最后24 tasks等权。首轮不加critic、pose
shaping、progress classifier或validation-derived reward。

## 5. Per-sample flow ratio与ASPO

rollout由cycle起点的old Writer产生。对每个executed prefix、每个 keyed
`(time, Gaussian noise)`样本，使用同一对随机数分别计算old/current CFM loss：

```text
rho_i = exp(clamp(loss_old_i - loss_current_i, -d, d))
```

每个action chunk使用`N_mc=4`个独立样本并分别形成ratio；不先平均loss再clip。正
advantage使用PPO clipped objective；负advantage使用SPO回拉项：

```text
psi_pos(rho, A) = min(rho*A, clip(rho, 1-eps, 1+eps)*A)
psi_neg(rho, A) = rho*A - |A|/(2*eps) * (rho-1)^2
```

采用`eps=0.05`、最多4个learning epochs、Writer AdamW peak LR=`1e-5`作为首个profile
候选；最终epoch数由A40真实wall/memory和ratio clipping/ESS机制证据在launch前封存，
不得由validation outcome调参。old loss、MC seeds、advantages在同一cycle各epoch固定，
每epoch完整遍历本rank全部task后只做一次全局24-task update。

这不是把functional training loss重新当rollout指标。它只用flow loss差近似
old/current action likelihood ratio，而优化符号和权重来自真实closed-loop advantage；
checkpoint选择仍只认严格配对rollout。

## 6. 信息墙与通用性

Writer输入仍严格为task language + exactly one action-hidden teacher video。reward phase：

- teacher action/proprio/pose/terminal不进入Writer或loss；
- source policy、normalization和public LoRA topology冻结；
- 只读24 train tasks的official binary reward；
- validation/test reward和actions均为0 reads；
- 不使用task ID作为Writer输入，task identity只在外部scheduler中建立task内baseline；
- 不做multi-video、LoRA平均、checkpoint融合或task adapter bank。

该训练只要求一个可评估CFM loss的flow policy和一个可微的condition-to-policy-parameter
映射。以后换环境、source policy或用真正critic advantage时，Writer结构和ratio目标
不依赖LIBERO语义桶或监督学习专用辅助loss。

## 7. 工程 owner 与退役

- `src/ember/writer/*`：原位恢复唯一v6 Writer；Target-Owned活动schema/config不可再载入。
- `src/ember/reward/*`：继续拥有official reset、trajectory与executed-prefix CFM；增加
  可选failure replay，不复制环境runner。
- `src/ember/rl_writer/*`：原位替换旧success-filtered loop、contract、checkpoint和
  inference；新增最多一个cohesive flow-credit math/replay owner。
- `scripts/train_rl_writer.py`和`scripts/evaluate_pi05.py`仍是唯一入口。
- 历史RL、Target-Owned与v6结果由Git/config/artifact保存，不留可执行兼容分支。

task-local RL仍是后续阶段，不随本设计自动启动；其历史success-filtered helper在本阶段
不扩展，等新RL-Writer成立后再决定是否迁移。

## 8. 验证和裁决

实现后按以下顺序推进：

1. CPU：v6 identity/freeze/shape、K4 advantage、success+failure replay、per-sample ratio、
   positive/negative gradient符号、24-task/world-size assignment、checkpoint fresh/resume；
2. live GPU：gpu01/gpu02实时空闲比较，总数不超过6；最长105-frame、最长failure
   trajectory、Nmc4、fresh/restore与两epoch profile；
3. cold start：fresh v6，直到预声明coverage exit或macro400上限；
4. reward：先2个outer cycles看nonzero task数、ratio、clip fraction、LoRA/action变化和
   throughput；机制成立才续到封存horizon；
5. paired rollout：稀疏checkpoint先64-state screen，候选才做8×50 correct；single
   checkpoint必须严格`>150/400`；
6. winner做correct/same/wrong/shuffled/reversed、breadth/churn、Core→Procedure→LoRA→
   action和SFT geometry对照。

失败分层：24-task reward coverage不足属于exploration/cold-start；advantage存在但
Writer梯度近零属于ratio/credit mechanics；train reward升而held correct下降属于
source overfitting；correct升但wrong同升属于video identifiability。四者不能混写成
同一“RL无效”。

## 9. BCI实现与AS profile seal（2026-08-04）

- canonical v6与task-relative reward path已原位实现；旧success-filtered Writer-RL、
  flat task-local RL和Target-Owned活动模块已删除，历史由Git/artifact保存。reward
  ledger可显式保留failure replay；flow-credit owner实现同task LOO advantage、逐CFM
  sample ratio、PPO正项、SPO负项和episode/task等权归一化。
- 六卡CPU/运行合同验证通过。聚焦RL/reward/evaluator为43项；全仓因同进程累计内存
  边界拆成135项与75项，两组均通过。architecture guard无hard violation；live Writer
  generation从评测authority解析中拆出cohesive owner，不形成第二个evaluator。
- `gpu02:1,2,3,4,5,7`的fresh AS profile保留logical B20、policy B2和16-frame encoder
  chunk。三步`33.464/30.886/30.977s`，峰值allocated/reserved
  `34,948,858,880/44,816,138,240` bytes；最长真实stride5视频105帧，0 OOM、0 clip。
  step1的factor-only梯度是template-A/zero-B staging；到step3 semantic frontend、Core、
  Program、compiler与factor全部finite/nonzero。
- 独立root先fresh0→1再exact-resume1→3，合同保持
  `1d2290eac6cd148a33f6f83dfeb006a97bcd68a9dfc4de1cf49704263d457a87`，metrics严格
  1/2/3，累计1,440 query与72 video conditions，validation/test action reads均0。
  AS config据此seal为fresh 0→400、every25，首段stop25；profile权重不得进入formal。
- AS macro25后直接用canonical reward cycle的K4 pre-update trajectories作coverage，
  避免再造第二套random-reset probe。若24 tasks未全部至少一次success，该reward run
  仅作候选probe，AS只从同一formal root exact-resume到下一25-step边界；若全过，同一
  cycle继续提供最长failure/Nmc4/two-epoch机制profile证据。

## 10. step25 reward profile裁决（2026-08-04）

- AS fresh0→25完成且健康，但pre-update K4仅12/24 tasks至少一次成功；25/96总成功、
  9 mixed、3 all-success、12 all-failure，因此按第6节预注册门不得冻结step25 cold start。
- 9 mixed tasks产生稳定非零credit：两epoch ratio范围`[.9860,1.0174]`与
  `[.8902,1.0629]`，positive clip fraction`0/.001781`，grad norm`.04016/.03035`。
  这支持estimator和正负credit实现可运行，不支持当前checkpoint已提升closed-loop。
- A40峰值reserved`45,183,139,840` bytes，现有K4/Nmc4/B2为资源上限附近的sealed
  profile；后续不增加K、Nmc或batch，只通过同一AS训练轴改善cold-start coverage。
- runtime根修不改变算法：每rank显式绑定sealed LIBERO assets，并把torchrun local rank
  映射到真实physical EGL device。有效run的24×4 ledger、failure retention、信息墙和
  physical GPU topology均完整。下一段为同一AS root exact-resume25→50。

## 11. step50 reward profile裁决与collective根修（2026-08-04）

- 同一fresh AS root exact-resume25→50后累计24,000 queries与1,200 one-video
  conditions，0 OOM/clip。K4 pre-update得到38/96 successes、14/24 task coverage、
  10 mixed、4 all-success、10 all-failure，仍未达到第6节24-task exit gate。
- 与step25的96个task/cursor严格共享env seed、初态hash、policy seed、teacher demo及
  共同policy-noise prefix；gained/lost/retained=`19/6/19`，success`25→38`、coverage
  `12→14`。这是真实净积累，但task5/16失去coverage且10个tasks仍全失败，不能将总体
  上升解释成漂移已解决。
- 首次step50 profile在rollout后暴露纯工程故障：不同rank的mixed-task数量为
  outcome-dependent，0-mixed快rank先enqueue NCCL gradient sum，而慢rank仍进行Nmc4
  本地反向；480秒watchdog终止进程。增加timeout或关闭watchdog会掩盖错误collective
  生命周期，不能作为修复。
- canonical修复在每个learning epoch完成本地反向后使用独立FileStore all-rank-ready，
  所有rank ready后才按同序进入NCCL sum。原六卡、96 rollout、两epoch规模重放中，
  96/96 rollout JSON与失败run字节级一致；两epoch均finite、完整cycle1 checkpoint、
  0 watchdog，证明只改变collective入场时序而没有改变scientific sample或objective。
- A40峰值reserved降至`40,342,913,024` bytes；两epoch ratio范围
  `[.9905,1.0094]`/`[.8555,1.0559]`、positive clip均0、grad norm
  `.02872/.02697`。coverage仍失败，因此该checkpoint仍只作profile，下一段只从AS
  step50 exact-resume到75。

## 12. step75 reward profile裁决（2026-08-04）

- 同一fresh AS root exact-resume50→75后累计36,000 queries与1,800 one-video
  conditions，75个full24 macro均finite，0 OOM/clip。K4 pre-update得到47/96
  successes、18/24 task coverage、13 mixed、5 all-success、6 all-failure，仍未达到
  第6节24-task exit gate。
- 与step50严格共享96个task/cursor、env seed、初态、policy seed、teacher demo及共同
  noise prefix；gained/lost/retained/both-fail=`21/12/26/37`，success`38→47`、coverage
  `14→18`。task`9/16/19/25/37`获得coverage而task4失去；这是比step50更强的净积累，
  但12次能力丢失和单task换手仍禁止“task drift已解决”的结论。
- suite success spatial/object/goal/libero10=`12/17/12/6`，coverage=`4/6/5/3`；object
  已全覆盖，剩余6个全失败task为`4/5/29/36/38/39`。因此不改变K、Nmc、batch或exit
  gate，只沿同一AS轴继续到100。
- 两epoch ratio范围=`[.97777,1.02659]`/`[.91171,1.08119]`，positive clip=
  `0/.000247`，grad norm=`.03184/.02709`，peak reserved=`40,340,815,872` bytes；rank-local
  mixed分布`3/1/4/1/3/1`下完成两轮finite update、完整cycle1 checkpoint和0 watchdog，
  为FileStore collective入场合同提供第二个原规模实证。
- AS resume首次因物理选卡产生`4+2` NUMA rank分布而与root封存的`3+3` topology不符，
  在训练前被合同正确拒绝；保持原`3+3` rank topology后完成。该事件不改变科研样本，
  也不构成重启或放宽合同的理由。下一段只从AS step75 exact-resume到100。

## 13. step100 coverage回落与条件写出审计（2026-08-04）

- 同一fresh AS root exact-resume75→100后累计48,000 queries、2,400 one-video conditions
  与100个finite full24 macros，0 OOM/clip；最后segment wall=`805.085s`。24 tasks均为
  2,000 queries、100 video visits和50/50 unique videos，A40适配没有减少训练数据量或
  optimizer宏步语义。
- K4 pre-update为52/96 successes、17/24 coverage、11 mixed、6 all-success、7
  all-failure。相对step75严格配对gained/lost/retained/both-fail=`14/9/38/35`，共同
  noise prefix 96/96一致；task20失去coverage且没有新task进入。success轨迹
  `25/38/47/52`虽上升，coverage`12/14/18/17`已否定breadth单调成熟解释。
- 两epoch ratio=`[.98452,1.00771]`/`[.88801,1.06045]`、positive clip均0、grad=
  `.02535/.02563`，两轮finite、完整cycle1 checkpoint和0 watchdog，说明credit机制
  可运行；但coverage exit失败，所以不启动正式RL，也不继承profile更新。
- owner随后明确：应优先理解Writer生成LoRA的rank、能量、条件差异与policy有效传递，
  RL可以使用但不能成为绕过LoRA质量问题的默认答案。因此在下一25-step AS segment前
  插入只读内部审计：全部24 train tasks固定demo0--4，比较step25/50/75/100的真实`BA`
  spectrum、rank-coordinate energy和same-task video variance；另按每suite首/尾train
  task预先固定8-task面板，用demo0 frame0 observation、相同policy noise测identity、
  other-demo、reversed与shuffled的fixed-action传递。该审计读取0条target action，
  不使用functional loss或validation/test结果，不改变cold-start checkpoint与数据合同。
- 审计与K4共同裁决下一步：若LoRA条件方向随AS成熟且action传递改善但coverage只是一次
  换手，可续同一AS轴；若有mixed reward却条件方向仍落不进action有效子空间，才进入
  Relative-Flow RL；若v6仍保持近rank1、高B-column相似与弱video传递，则优先v5.2同credit
  对照或重构condition-to-policy存储/组合。禁止仅凭aggregate52或健康ratio选择RL。

## 14. 四点LoRA/视频到action审计与下一裁决（2026-08-04）

- clean/pushed`2b775f0`在6 ranks上完成step25/50/75/100×24 train tasks的96-row正式
  审计，wall`291.333s`、peak reserved`19,308,478,464` bytes，0 target action与0
  validation/test reads。全部task固定demo0--4；fixed-action面板严格为每suite首/尾
  train task，不按outcome选task。
- 四点effective norm中位=`53.40/80.37/93.17/99.18`；stable rank=
  `1.000028/1.000055/1.000153/1.000176`，top singular share=
  `.999972/.999945/.999848/.999825`，q/v B-column cosine始终约`.999`。这复现历史v6
  rank collapse，但不是新根因：direct SFT stable rank约1.52仍偏低，Target-Spectral
  强制升到3.32却correct34，禁止再次以谱/正交作为优化目标。
- same-task五video centered/sample energy中位=`.0813%/.1309%/.1333%/.1300%`，demo1
  BA relative-L2=`.0380/.0506/.0499/.0475`；step50后均平台。fixed action的same-video
  relative-L2=`.0081/.0071/.0094/.0101`，而reversed=`.0299/.0508/.0606/.0498`、
  shuffled=`.0216/.0325/.0351/.0474`。因此Visual Transition/Procedure/order路径真实
  工作，最弱的是video instance条件差异的闭环方向，不是整个video入口断路。
- step100 covered与all-failure tasks的norm中位=`99.13/102.37`、centered video energy=
  `.1290%/.1311%`、demo1 BA差异=`.0405/.0634`；Spearman与success分别约
  `-.09/-.22/-.22`。失败task的LoRA更大、video变化也更大，直接否定“加scale、增rank
  或扩大条件差异即可成功”。相邻checkpoint BA/action churn中位从step50
  `1.116/.187`降至step100`.608/.142`，但仍远未稳定。
- 当前结构已具有可用的task主写入与时序因果链，却没有closed-loop reward来区分有用与
  无用条件方向；这正是Relative-Flow credit要检验的接口。由于step100仍有7个all-fail
  tasks，task-local LOO在它们上没有credit，不能违规跳过coverage门。下一段只续同一AS
  100→125后重做K4；若coverage支持则进入正式reward，若继续停滞再裁决v5.2同credit或
  更根本的credit/exploration重构，而不是另做rank/scale/store小改版。

## 15. AS125与binary-only Flow-Credit负裁决（2026-08-05）

- 同一fresh AS root exact-resume100→125后累计60,000 queries、3,000 one-video
  conditions和125个finite full24 macros；step125 checkpoint完整，0 OOM/clip与0
  validation/test action reads。首次resume因漏传sealed`--num-workers 0`在step101前
  fail-close且未写训练状态，补齐完整CLI后原root正常完成。
- step125 K4为50/96 successes、19/24 coverage、14 mixed、5 all-success、5
  all-failure。相对step100严格配对gained/lost/retained/both-fail=`10/12/40/34`；新增
  coverage为task5/29，持续全失败为task4/20/36/38/39。task36/38/39在
  step25/50/75/100/125全部0/4，因此24-task binary reward support不是靠继续同一AS
  就可合理预期获得。
- profile仍完成两轮真实credit并封存：ratio=`[.98710,1.01237]`与
  `[.76458,1.10147]`、grad norm=`.03615/.05310`、clip均0、peak reserved
  `40,338,718,720` bytes，完整cycle1 checkpoint、0 watchdog/OOM/nonfinite。该证据只
  验证实现；cycle1 Writer权重继续禁止继承。
- step100/125的48-row内部审计显示norm中位`99.18→109.11`，但video energy
  `.1300%→.1154%`、demo1 BA差异`.0475→.0448`、fixed-action demo差异
  `.0101→.0086`均未增强；BA/action churn为`.536/.138`。全24任务success变化与
  video-energy变化Spearman=`-.521,p=.0090`，与BA churn=`.416,p=.0430`。持续全失败
  组的video energy、demo间BA差异和churn均不小，进一步否定rank/scale/差异放大解释。
- 因此本设计的binary-only形式正式负裁决，不启动formal outer cycles，也不继续AS150。
  下一方法必须在不读teacher action、privileged state或任务特化规则的前提下，让同为
  binary failure的trajectory按teacher-video所示语义状态变化获得相对credit；binary
  success保持最高优先级。历史v4已证明normalized-video-progress易退化为时钟，后续
  设计不得把teacher帧ordinal或rollout长度本身作为value。
