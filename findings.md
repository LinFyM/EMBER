# EMBER Findings

更新时间：2026-08-13。本文只保留跨架构仍成立的第一性原理结论。逐方法结果、精确旧 commit/root 与禁止重复项
见`docs/research_history.md`；逐日原始记录可由 Git commit`3a6f801`读取。

## 1. Current empirical boundary

长期目标是同一 shared method、同一 single checkpoint 的 strict paired correct 严格`>150/400`。目前未达到：

2026-08-13新的active架构判断是Dynamic-K backbone-memory rank-8 Writer。它不是从旧rank16 checkpoint压缩，
而是针对same-task不同video correction近正交与post-backbone compiler衰减，从真实图文+Action Expert联合forward
直接形成逐层memory grid；只让有符号视频变化进入dynamic value，再由跨video共识和共享layer/rank mapper写出
fresh完整LoRA。它继承v6的task-grounded address、真实frame ordinal与causal Procedure，但不继承旧K4的固定
四基/DCT16/phase-wise mean，也不继承已退役Program residual/guard/solver。

实现与live机制门已经闭合。无frame预算的真实A40 profile在micro10 OOM；micro8虽然不OOM，但完整K-set的
joint backbone一宏超过16分钟并触发480秒NCCL heartbeat，故该执行图不可用。唯一工程修正是在stride5候选后、
joint backbone前给每condition固定64个真实有序frames，并对K条video使用相同`floor(64/K)` cap和含首尾均匀
索引。budget64 world6 full24 B20 profile一宏`32.8066s`，K1/2/3/4各6 tasks，functional loss`.15611`、
consistency`.009984`、gradient norm`.02361`均finite，峰值allocated/reserved约`39.15/45.41GB`；checkpoint完整。
这只证明执行图、动态K和梯度链路可训练，不证明closed-loop有效，下一裁决是fresh macro50 strict paired400。

| method | correct | same | wrong | shuffled | reversed | interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| v5.2 old | 132 | 138 | 74 | 82 | 83 | 视频特异性强，absolute不足 |
| v5.2 task-complete | 120 | 109 | 107 | 111 | 124 | recipe削弱Procedure和absolute |
| v6 old | 121 | 122 | 111 | 84 | 47 | 顺序影响强，但absolute低、旋转大 |
| v6-fast task-complete | 143 | 135 | 125 | 128 | 129 | 历史最好single checkpoint，视频margin弱 |

最新 uniform pivot-rank14 裁决：

| arm | correct | breadth | relative to old134 retained/gained/lost |
| --- | ---: | ---: | ---: |
| immutable old full-rank | 134 | 6 | reference |
| compiler-only old-cache transform | 138 | 7 | `119/19/15` |
| online-regenerated rank14 | 128 | 7 | `113/15/21` |

compiler-only净增4，却违反预注册 lost`<=10`；增益由 Long1 净`+11`掩盖 Spatial/Object 净`-3/-4`。从
compiler 到 online 又有`13 gained/23 lost`。因此 uniform compression 与 online regeneration 都会造成
独立的 target-heterogeneous capability rotation。rank14 路线、Gate C、cycle1、controls和训练已经退役。

最新MGCI-JC裁决：full96 mechanism profile把PVJFC/CGIK condition从`597.861/270.188`修到`174.813`，并保持
rank`48/96`、negative ratio`.02088`及Program→BA→action非零；但formal macro5 strict只有`134/400`、breadth6、
per-task=`1/5/46/30/0/34/18/0`。相对old134为`114 retained/20 gained/20 lost`、churn40，suite净值=
`+1/-6/-1/+6`。因此健康key、rank和transmission仍与能力换手共存；当前最早失效接口是blind offline B20
source-action cotangent RHS到held on-policy useful support的映射，而不是condition谱或幅度不足。当前没有active
successor，后继credit方向等待owner讨论。

## 2. The actual learning problem

EMBER不是从视频复制一段动作。部署时策略面对未见初始化，Writer必须从语言与 action-hidden video 提取跨
轨迹仍成立的对象、关系、目标和有向过程，再把这些信息编译成能改变 frozen policy 行为的一套完整 LoRA。

监督训练故意让 video 与 action query 同 task、跨 episode。这样阻断逐帧动作复制捷径，并要求一套 LoRA
覆盖不同初态；但它也造成核心不可识别性：同一 task 的 action target 对不同 teacher video 在统计上相同，
所以普通 positive functional loss 可以只学 task controller，而无需解释哪段视频证据、哪个时间顺序使该
controller成立。架构必须让视频信息成为生成 LoRA 的必要动态值，最终仍要由严格 controls 证明它被正确使用。

## 3. Video use versus correct video understanding

历史 v4 已证明 shuffled/reversed 会改变 latent、LoRA 和 action；但其 shuffled 成功甚至高于 correct。原因不是
“模型没看视频”，而是它把 absolute frame phase、translation trajectory 等低层相关性编译成 controller。

因此有四个不同命题：

1. 视频改变 hidden state；
2. 视频改变 effective LoRA；
3. 视频改变 policy action；
4. 正确视频的语义与顺序产生比 wrong/shuffled/reversed 更有用的闭环行为。

只有第4项支持“从教学视频学到任务知识”。normal order 是训练分布并不自动赋予它因果语义；shuffle/reverse
必须在真实输入帧上完整重排，保持 task/state/RNG/video identity 配对。correct 需要沿有用 policy direction
胜过 negative，而不是仅靠把 negative 推坏形成 margin。

## 4. Why task drift is the central optimization symptom

EMBER多次出现某些 task 在一个 checkpoint 新增、另一些同时丢失。SFB 的八 checkpoint success union 是193，
single best却只有127，证明能力存在于训练轨迹各处，却没有稳定共存于同一参数状态。

已被分别观测到的换手来源包括：

- 不同 task、query、flow noise 的 functional gradients 近正交或相互冲突；
- 同 task 不同 teacher video 的 correction 在 effective LoRA 空间近正交；
- shared factor、Reader、condition map 或 Program→factor compiler 把异质 cotangent 压成 common update；
- 离线 feature-row/RLS 保留不等于 held on-policy state occupancy 保留；
- 强制压缩、refactorization与 online regeneration 即便误差很小，也会跨 closed-loop decision boundary；
- optimizer在轮流获取能力，而非形成可重复的共同累积。

所以“训练更久”“loss继续下降”或“checkpoint union很高”都不能解决 drift。每个新设计必须解释它为什么能让
不同 task/video 的有用更新在同一 checkpoint 共存，并用 paired gained/lost/churn验证。

## 5. LoRA health is a diagnostic, not the objective

正常 SFT LoRA 提供有价值的 policy geometry 参照：能量不能小到近 identity，target间结构不能完全同质，
有效 rank 与 coordinate participation 也不应无故坍缩。但历史结果同时否定两种极端：

- 过低能量、近 rank1、target cosine接近1常解释写出端容量不足；
- 人工追求均匀谱、高 stable rank、正交、更多 atom/lane/expert 或 SFT量级 norm，也反复得到更差 closed-loop。

policy真正需要的几何可能天然 q-dominant、低 effective rank、跨层 coherent。raw A/B 还有 gauge ambiguity，
因此正式比较优先用 effective `BA`、fixed-action response与closed-loop，而不是因子符号或漂亮谱形。

Reward-Credit进一步发现：continuous q/v tangent约`1e-8 RMS`，低于非零 native BF16 factor约`1e-4`的局部
ULP；但随后 uniform rank14 为 tangent 预留 physical slots 又损伤 old support。这说明数值瓶颈真实存在，
却不能靠牺牲已有 policy manifold 来修；更不能为了低位重现降低 batch、dtype效率或GPU利用率。

## 6. Functional objectives are weak surrogates

source-action SmoothL1、reconstruction、expert cosine、gradient consistency、MC variance、tube radius与row-wise
retention都曾显著改善，同时 strict closed-loop不升反降。原因是离线 action queries只覆盖局部功能切片，而
rollout会改变状态分布并跨越离散成功阈值；tiny parameter error也可能落在高敏感 policy方向。

证据层级固定为：

1. shape/freeze/resume/finite/closure：工程合同；
2. representation→compiler→effective BA→fixed-action：机制证据；
3. real strict paired400：方法裁决；
4. five/six-arm controls：视频因果解释。

前两层只能决定是否值得跑下一层，不能选择最终checkpoint。80-row screen也曾与full400对保留性给出相反
结论；正式选择不得依赖小panel。

## 7. What task experts solve and do not solve

正式24-task expert bank统一续到step2000。direct-expert train closed-loop从step250到2000为
`432/557/624/638/658` of 1200，step2000有23/24 tasks非零。它证明 task-local SFT LoRA 是
policy-effective target，并提供能量、rank coordinate与跨target结构参考。

它不解决：

- held task 泛化；
- 同 task 不同视频的特异性；
- 正确、乱序、倒序视频的时间因果；
- shared Writer在24 tasks上的稳定共同积累。

同一 task 的 expert reconstruction target 对所有视频恒定，所以只用 reconstruction 训练最多学到
task-level parameter manifold。soft/hard 24-expert bank在held只得`15/80`与`3/80`，又说明一个可近精确重建
的训练task dictionary不能直接成为deployment route。expert只能作为train24 privileged teacher或几何依据，
不能成为held输入、nearest-neighbor oracle或第二套LoRA。

## 8. What few-shot may solve and cannot be assumed to solve

K4历史实验显示，多视频集合可以削弱单条demo的偶然速度、视角和轨迹细节，并改善permutation、same-video与
leave-one-out内部稳定性。这支持用户的直觉：跨demo不变量可能更接近高层任务知识。

但K4 best只有108，full24 gradient retention约`.043`，并未解决正确顺序的policy-effective credit或
single-checkpoint drift。简单平均video features、平均LoRA或让数量越多越好，都可能抹掉有向过程或只增加
计算。未来few-shot必须作为独立matched变量：固定`k`或显式定义variable-cardinality set encoder，保留每条
视频内部时序，禁止挑video，和相同训练/eval预算的one-shot对照。它不是当前已证明的捷径。

## 9. Coherent historical causal chain

1. v4暴露错误的absolute-time/action-phase shortcut。
2. v5/v5.2分离Semantic Core与Procedure，增强wrong-video辨识，但时序在compiler端衰减。
3. v6-fast把absolute推到143，也暴露架构与task-complete recipe强耦合及晚期换手。
4. v7/v8/v10与Loom/Core/Prior说明更漂亮内部时序、fusion或去DC不保证policy-effective方向。
5. Target-Spectral、atom/lane/expert路线说明强制“健康几何”不能替代闭环。
6. Target-Bound证明动态路径能工作；SFB的union193证明核心是共享共存与稳定积累。
7. variance reduction证明functional estimator不是首因。
8. few-shot与video-grounded experts证明视频敏感、参数隔离和集合稳定性仍非充分条件。
9. task experts给出有效task target，但Expert-Manifold routing/reconstruction未带来held support。
10. Balanced residual、RLS、Reward-Credit依次定位跨video正交、offline/on-policy错位和native-factor ULP。
11. 最终rank14去混杂证明compression与regeneration都会独立破坏能力，不能再用微小数值补丁救路线。

这条链要求下一设计延续已有认知，只改变最早失效接口。负结果只淘汰实际受检验的假设：rank14失败不等于
所有rank reservation失败，expert bank失败也不等于所有task-level manifold监督失败。

## 10. Latest interface verdict

PICK已经给出一次清晰的接口定位：frozen-policy innovation的raw same/order结构、full48 correct/negative motion、
Program→BA→action和吞吐均工作，但48-key regularized Gram condition=`483.61515>200`。因此不能把它写成方法
成功，也不能把失败归因于video encoder、compiler或动作增益；PICK未进入formal训练。

PICK-GC只把PICK的static block换成terminal goal residual，保留centered causal prefix。
train24 cache的same/cross/reverse与condition门，以及exact target-language wrong full48均已通过：raw rank48、
condition=`152.45803`；world6 mechanism condition=`152.61008`、retained/null=`24/24`、Program→BA→action闭合。
B32 deployment又证明zero-memory native LoRA/action bit-exact并完成canonical cache-to-rollout。因此PICK原先的
common-mode conditioning接口已经被这一单变量修复，不能再把后续失败归因于key数值不可解或compiler无增益。

world4 formal fresh`0→10`已完成：每步feature rank48，condition稳定可解，最终FP32 Program memory RMS
`3.5493e-6`且全部值非零。strict paired400只有`138`、breadth6；相对macro0 retained/gained/lost=
`118/20/16`。effective-BA整体范数中位比`1.000016`、相对L2`.002397`，说明小而真实的policy tangent已经传出，
却仍造成36个episode换手。故PICK-GC+blind offline credit按预注册门退役；失败不能再归于condition数值、
identity写出、LoRA能量或单纯compiler闭合。

当前最早科学失效接口是blind train24 source-action functional cotangent到held on-policy有用support与共同积累的
映射。OSG-PC曾不再用微小reward cotangent替换proposal，而是对每条成功train24 rollout构造
executed-prefix Program half-space，把原B20 proposal解析投影到“不一阶损伤已有success support”的可行锥；
full48 key、FP32 Program和native rank16保持。它已完成CPU实现但未获得完整live mechanism evidence；heterogeneous
topology、few-shot和task-level manifold supervision仍是开放问题而非自动fallback。

OSG-PC现已完成唯一canonical CPU实现：success binary只选择保护约束，失败轨迹完全不进入replay/gradient；
每条成功episode形成独立Program cotangent，最多4条约束由解析KKT投影，no-success与raw-feasible逐元素退化为
blind proposal。历史完整K4 flow panel只按成功row ordinal取样，避免success-only压缩改变另一条guard的MC身份。
full48仍是唯一shared write owner，并新增continuous/native实际应用后的guard/source-descent报告，因为task-local
可行不自动保证shared damped solve后的真实memory motion仍可行。fresh完整CPU回归`340 passed`只关闭工程接口；
唯一world6 live profile在full48前失败：rank5在`profile_max_seconds`等待600s后watchdog，run-contract到timeout
至少`969.9709s`，相对matched baseline至少`1.912x>1.25x`。此前11个collective正常，CPU rank3 B20 loader仅需
约2.75s总计，说明current K4 rollout/retention task path产生不可接受的rank-local长尾；但因没有stage journal，
不能继续把它归因到具体simulator或VJP。没有task records、projection或shared guard report，因此科学guard假设
仍未被正反验证；被淘汰的是current full-replay per-success VJP执行图，而不是所有on-policy success constraint。

下一单变量authority是SKNC。它保留PICK-GC的ordered goal-causal key、B20 blind cotangent、FP32 Program与
native rank16 compiler，但K4只保留binary outcome：current correct-video LoRA只有在四条random-reset lanes
`4/4`成功时，其condition key才进入硬约束。24-task full48 objective仍完整求解，memory增量被参数化到current
4/4 keys与每train task第一条persisted 4/4 key共同span的正交补，因此不是按task乘零的scalar gate。

该约束直接使`phi_success @ DeltaMemory=0`，目标是让完整conditioned Program/LoRA/policy函数在shared write后
保持不变，而不是像RLS只保护offline action rows，或像OSG-PC回放某些成功prefix的一阶loss。它删除所有成功
trajectory replay、CFM VJP和per-success task-local cone，成本只剩同源K4 rollout加small-matrix solve。其关键
风险也已预注册：anchor span可能压平可学习feature，single train24 success key可能不能代表same-task其它video
或held occupancy。canonical实现、fresh schema和完整CPU回归`334 passed`现已关闭工程接口，但尚无live或性能
证据；zero anchor motion只能是机制门，最终仍由macro5/10的strict paired400 retained/gained/lost与视频controls
裁决。

首个SKNC world3 live profile进一步给出强机制证据：11个4/4 anchors覆盖四suite，rank=`48→37`、condition=
`29.65`、projected energy=`.778`，outcome-only、negative null、unprotected descent、LoRA/BA/action和吞吐
全部过门。唯一false是Program ratio=`1.1228e-4`；但protected LoRA/BA/action全为exact zero，且同stored update
的GPU full-FP32 ratio=`7.10e-8`、TF32=`8.44e-5`。因此这是hard-equality diagnostic错误继承production TF32
所致，不是success-key nullspace科学假设失败；首root仍作为measurement non-pass保留，只允许修测量后重过。

clean `f4fdac7` reprofile保持同一outcome panel与所有科学变量，success/failure仍为`61/35`、11个4/4 anchors；
关闭TF32的既有constraint diagnostic把Program ratio恢复为`8.95e-8`，16/16 checks全部通过。rank=`48→37`、
condition=`29.65`、projected energy=`.778`、protected LoRA/BA/action exact zero、step=`478.627s`和scaled wall
ratio=`.47173`共同说明SKNC的representation→constrained Program→native compiler→fixed action接口已接通，
但仍不能推出closed-loop改善。实际deployment panel的B8/16/32均stable，B32最高`.47166 LoRA/s`，且没有
hidden teacher read；因此工程接口关闭，最早未决接口重新落到blind B20在success-key nullspace内能否产生
跨task共同的held on-policy改善，必须由fresh macro5 strict paired400裁决。

formal fresh`0→5`给出了清晰的负答案。训练在world3完成，persisted anchor bank从11增至15 tasks；macro5仍有
rank36、projected energy`.59195`、protected/unprotected Program ratio=`1.276e-7`，且没有hidden read、replay
或reward gradient，故不是nullspace容量坍缩或compiler重新失联。strict paired400为`137/400`、breadth7、
per-task=`1/3/45/32/0/37/18/1`。相对old134严格配对为`121 retained/16 gained/13 lost`、churn29：Long task1
净`+7`和Goal6净`+2`伴随两个Object tasks净`-5`、Spatial3净`-2`。SKNC把PICK-GC的lost16降到13，但仍远高于
预注册8，且总分未过140；hard train24 success-key protection没有外推为held video/初始化support protection。

因此最早失败接口进一步收窄为两部分的交界：PICK condition key对单条train video的完整LoRA地址是可保护的，
但它不是跨video/occupancy的任务support坐标；同时blind B20 source-action cotangent在剩余nullspace内没有真实
reward方向保证，继续发生suite换手。这个结果只淘汰“PICK-GC + first-all-success-key nullspace + blind B20”，
不否定所有binary success constraint、on-policy continuous credit、few-shot或task-level manifold supervision。

下一单变量authority据此选择SRTP。它不再把reward tangent作为sub-ULP direct update，也不恢复OSG每条成功
trajectory的full-prefix VJP；SKNC先产生完全相同的shared `D0`，mixed K4 tasks再用first/last加两个reservoir
interior landmarks形成每task一个LOO signed Program cotangent。约束`<r_i, phi_i D><=0`直接在24-task汇合后的
最终memory上同时求最近投影，all-success仍由anchor equality保护、all-failure仍保留B20 acquisition。历史同源
Reward profile的11 mixed tasks原需4452 chunks/928 forwards；固定16 rows/task与Nmc4只需44 forwards，针对性移除
OSG长尾而不减少K4 outcome覆盖。这个设计的最快否决证据是raw constraints不冲突、投影近零、negative motion
破坏或wall超过matched SKNC `1.25x`；只有live shared projection接通后才允许fresh macro5 paired400。

SRTP的canonical实现曾通过CPU工程门。rollout在线reservoir保证每episode最多4 rows并保留真实
first/last；mixed panel只做4次logical B<=16 CFM forward，homogeneous panel严格0 forward；最终投影用anchor-
null feature与Program cotangent的factorized Gram在CPU FP64 NNLS求dual，只在GPU FP32合成一次大correction。
synthetic覆盖了无约束/原更新已可行的逐元素退化、相关与重复half-space、task permutation、KKT feasibility、
anchor closure和正alignment；50组额外随机相关约束也无violation。fresh checkpoint schema拒绝SKNC resume，
ephemeral landmark不进TaskObjective record或checkpoint。完整CPU回归`359 passed`不能预测closed-loop；原计划
只有在mixed>=8且至少3 suite覆盖、四suite all-success anchor、raw violation>=2、final violation=0、
energy>=.25、negative ratio<=.15和matched SKNC wall<=1.25x全过后才允许formal，实际运行在这些机制证据产生前
就因OOM停止。

首个`d172add` world3 live macro没有产生科学机制结果：三rank均在mixed reward CFM处OOM。它先把最早工程失效
定位为decoder graph lifetime；针对性修复让blind VJP立即消费原graph，reward LoRA cotangent完成后只重解同一
Program compiler一次，视频、condition、policy forward、objective、seed和batch均不变，完整CPU回归`359 passed`。
但唯一`e31e2fd`同合同reprofile仍三rank在相同CFM路径OOM，申请`254/484/484 MiB`时只余
`19/16.31/417.06 MiB`。因此更早且充分的失败接口是完整logical B<=16 landmark policy-gradient本体显存，而非
decoder graph；SRTP执行合同终局退役，不降batch/dtype、不加allocator开关、不做第三次修补。由于两次都没到
shared projection report，这个结果既不证明也不否定reward half-space的科学价值，只否定当前获取它的执行图。

SRTP之后的单变量选择是PCUG。它不再估计高维reward gradient，而是先形成真正准备commit的blind shared `D0`，
再对两个相同random initializations严格配对运行base与candidate Programs。只有candidate造成的paired losses严格
多于gains时，该task才在最终write获得临时zero-motion guard；base/candidate四臂全成功仍进入persistent
first-success bank。final update是`D0`到这些keys共同nullspace的closest projection，不重算full48，因此guard
不会再被shared solve改写，其他tasks也只发生必要的最小变化。每task仍总计4 rollouts且reward路径无policy
backward。它最快会被以下证据否决：candidate与base几乎无discordance、没有跨suite harmful tasks、projection
退化或破坏rank/energy、wall超过matched SKNC `1.5x`，以及macro5 strict不能把lost压到8以内并提高absolute。

PCUG canonical实现完成并通过完整CPU回归`344 passed`，但唯一world4 discarded macro在paired probe前就终局
non-pass。run-contract后`809.72185s`仍未完成Phase A full24 gather，wall下界为scaled matched SKNC的
`2.25568x>1.5x`；物理3--5持续100%，物理6先等待，因此共驻在物理6的低利用率服务不是长尾来源。run无OOM、
nonfinite、mechanism report或checkpoint。这个结果只淘汰“static cost-balanced world4 Phase A + PCUG”执行合同，
既没有支持也没有否定paired candidate guard；下一设计必须先消除global candidate形成前的rank-local tail，不能
把未执行的科学机制写成负结果，也不能因此转向无关架构。

Work-Queue PCUG据此在当时成为唯一active successor design。旧sampler所谓dynamic只是在macro前按correct-video
sampled-frame cost做uneven static ownership；world4四rank cost仅`207/216/206/204`，实际却由一个rank先等待、
三个rank把Phase-A wall拖到`>809s`，说明该proxy不能控制critical path。B20 rows、video、counterfactual与policy
RNG本来都是`(task_id, task_visit)`纯函数，因此可以让空闲rank完成一个task后原子领取下一个，而不改变任何
科学样本、full24权重或PCUG update。每rank最多保留8个graphs，复用SKNC world3已证明可容纳的显存上界；
full48与paired rows仍按ordinal排序。这个设计只会被task-level live timing、Phase-A wall与随后原PCUG机制门
裁决，不用静态latency拟合、world-size sweep或更多防御性检查救失败执行图。

该执行接口现已完成canonical实现并通过完整CPU`345 passed`。新sampler public入口对world3/4/5旧schedule逐task
逐row保持完全相同B20；host-local cursor只做24次短`flock` claim，rank在每个真实GPU task边界同步后领取下一项，
因此ownership由实际完成速度而非CPU enqueue或静态frame proxy决定。full48 payload在原tensor gather内携带24条
小timing rows，支持0--8个local tasks而不新增object collective；Phase-A wall失败会在96条paired rollouts前写
non-pass并结束。旧PCUG schema/config状态与eval family已原位替换，旧结果只由Git、design和failure artifact保存。

首次Work-Queue PCUG world3 live Phase-A提供了两个分离结论。科学计算本体在`72.9700s`完成，只有matched
SKNC的`0.15246x`，24 tasks与三rank各8个ownership均完整，说明task-addressable B20与variable gather没有重现
旧static PCUG的`>809s`长尾；但claim累计`60.8736s`，来自两个约`30.4s`的`flock`等待。实现检查确认cursor位于
共享`/data1` output root，而design明确要求host-local cursor，因此这是storage/locking层工程违约，不是PCUG
科学non-pass。paired probe未启动、没有checkpoint。唯一窄修是把一字节cursor移到节点本地`/tmp`；架构、
objective、B20、world3、queue order、cap与全部hard gates保持不变后重跑原复现。

host-local修复后的WQ-PCUG完整profile把paired candidate从推测变成了实证。Phase A=`44.74125s`、total=
`558.05862s / 1.16596x`，48 exact pairs的base/candidate successes=`34/33`，有7 discordance、3 gains/4 losses、
3 harmful tasks跨Object/Goal；15-row correct guard保留`.76492`能量、rank33，Program/LoRA/fixed-action protected
closure均为零。这否定“candidate太小、paired reward无内容或correct guard不可行”解释，也保留了work queue与actual
candidate pairing机制。

唯一失败揭示constraint composition断层。full48 blind solve报告negative/unprotected ratio=`.03991`，但final
projection只对persisted/stable/harmful correct rows做homogeneous nullspace，最终negative ratio升到`.50179`且
24/24 tasks失败。因此先前negative zero-RHS不是没有作用，而是在reward-derived correct guard阶段被覆盖。
最小后继不是把`[G;N]`整体压缩update，而是解negative-preserving affine correction：`C=min ||C||`且`NC=0`、
`G(D0+C)=0`。它保留`ND1=ND0`，只改变final correction subspace，并避免把full48 support无必要压到约9维。

NPCG canonical实现把上述仿射问题化为两个小feature SVD：先取current negative rowspace，再在其正交补内求guard
minimum correction；large `D0`与correction仍为FP32。synthetic tests同时验证`G D1=0`、`N(D1-D0)=0`、
no-guard identity和duplicate-row不变性；profile把negative preservation设为独立hard gate，formal则在每macro
写Program前检查guard/negative closure、rank、energy、alignment与blind negative ratio。完整CPU`345 passed`，
没有额外policy/video forward、negative rollout、parallel trainer或防御性hash。

首个NPCG live profile确认仿射composition本身解决了WQ的最早科学断层：final negative ratio从WQ的`.50179`
降到`.03524`，wrong/shuffled/reversed均从`0/8`恢复到`8/8`，同时paired outcomes完全不变、rank33、energy
`.35360`且LoRA/effective BA/fixed-action closure健康。唯一失败是protected Program ratio=`1.5831e-4>1e-5`，
绝对残差只有`3.43e-10`。实现中constraint right-hand side与correction沿用了全局TF32，而真实Program read为了
equality诊断显式关闭TF32；两种数值语义错配直接解释该残差。因此首个run是工程合同违约，不淘汰NPCG科学假设；
唯一允许窄修是full-FP32 constraint matmul加一次固定residual refinement，不改门限或其它科学变量。

窄修后的matched reprofile给出完整机制pass，而不是通过改门：protected Program ratio降到`5.7508e-8`，final
negative ratio保持`.03524`且三类各`8/8`，paired outcomes逐项不变；rank33、energy`.35360`与全部LoRA/BA/action
证据仍健康，总耗时`554.99255s / 1.15955x`。这确认NPCG同时保留correct reward guard与blind negative suppression，
并把最早失效接口推进到真实多macro闭环积累。B8/16/32部署均稳定且B32以`.47144 LoRA/s`略胜；该差异只用于
吞吐选择，不构成科学性能证据。下一次有信息量的裁决是formal fresh macro5后的strict paired400。

NPCG formal fresh五宏首次把机制证据推进到task-complete连续积累：first-stable bank为`12/13/15/15/16`，每宏
guard restricted solve均可行，feature rank=`33/36/35/34/34`，energy=`.354/.487/.457/.320/.390`，blind
negative ratio=`.040/.026/.015/.014/.032`；五宏约束残差均约`1e-13`且step wall稳定在`531--565s`。这否定
“negative-preserving correction会随bank增长立刻不可行或压缩掉shared update”的快速反例，但仍不能证明
closed-loop support真正保留。

macro5 strict400最终只有`135/400`、breadth5；相对old134为`117 retained/18 gained/17 lost`、churn35，Long1
同一task内也出现10 gains/7 losses。因为五宏Program/negative closure、rank与energy持续健康，最早失败不在
solver或compiler。train24×50 action-hidden audit进一步显示first-stable key到同任务其它videos的正交残差
均值/p90=`.40954/.59171`；所有observed stable rows仍留下均值约`.27650`。因此精确保护train视频点不能代替
跨视频/held-occupancy support neighborhood，NPCG + blind B20正式退役。后继不能靠增加point数、平均key或
参数小扫，而应让每次shared write对same-task action-hidden video nuisance具有结构化等变/不变约束。

这个接口的最小结构性干预是CVEG，而不是增加point guards或改成few-shot部署。每task每macro保留原primary
one-shot condition，再从同一no-replacement cycle取一条action-query-disjoint的ordered companion，令
`E=phi(companion)-phi(primary)`。blind update位于`Null([persisted success keys; E])`，final NPCG correction
位于`Null([negative rows; E])`，所以每次新增shared Program motion对两条正确video相同，同时reward guard不能
重新打开negative或跨video nuisance response。retained cache的50-panel审计显示单companion后correct/reverse
过程能量中位数仍为`.780/.789`；2--4 companions会把correct能量进一步降到`.639/.529/.436`，首个版本因此固定
一条且不扫K。它只在训练期增加等变约束，部署仍严格一条action-hidden video生成一套LoRA。

CVEG canonical实现现已完成。primary schedule逐visit保持不变，companion只在训练图内编码一次；full24 gather
按ordinal携带`correct/negative/companion/cotangent`，blind anchors为`[persisted;E]`，final affine correction
保护`[N;E]` response。profile同时报告blind/final等变closure、E rank、correct/reversed过程能量、negative、
Program、LoRA、effective-BA与fixed-action传递；checkpoint cursor只封存pair schedule，不保存companion feature或
第二套LoRA。fresh schema拒绝NPCG exact-resume，eval deployment仍只接受一条video。针对性39项与完整CPU
`346 passed`，没有新增policy forward、hash、逐tensor扫描或parallel executable path以外的科学所需单条
companion video forward。下一裁决必须是clean pushed commit上的唯一discarded macro0 live profile。

CVEG首个clean world3 macro0完整执行，Phase A=`53.6446s`、total=`549.0601s / 1.1476x` matched SKNC；24
primary/companion distinct且action-query-disjoint，E rank24，correct/reversed过程能量中位数`.78197/.77227`，
negative final ratio`.03286`且wrong/shuffled/reversed各8/8，guard rank15、projected rank33、energy`.35404`，
96条paired rollout有7 discordance、2 gains/5 losses，Program/LoRA/BA/action与wall门均通过。22项中唯一false
是cross-video equivariance ratio：blind/final=`8.86e-5/1.26e-4`，但绝对E motion都只有`1.697e-10`，final
correction在E上的motion RMS=`1.11e-13`且violation0。说明final response-preserving solver正确保持了blind输入，
而blind FP64 rowspace投影转为FP32 21M-value write时留下TF32量级residual。该root是measurement non-pass，不是
架构科学non-pass；唯一窄修是复用NPCG语义，在原FP32 write上做一次fixed full-FP32 anchor residual refinement，
不改E、sample、ridge、scale、rank、dtype、forward或threshold，随后只重跑同一profile。

fixed full-FP32 residual refinement后的clean `2eb1e8e` matched reprofile把数值接口闭合且未改变科学面板：
22/22 checks全过，blind/final equivariance ratio从`8.86e-5/1.26e-4`降到`7.91e-8/9.56e-8`，E rank24、
correct/reverse energy`.78197/.77227`、negative ratio`.03285`与三类8/8均保持；base/candidate仍为`34/31`、
2 gains/5 losses。一次FP32 refinement使total从`549.060s`增到`584.649s`，仍为matched SKNC的`1.2215x<1.5x`。
这确认最早违约只是大Program数值落地，并未靠放门或改变update获得pass。实际one-shot部署B8/16/32 LoRA/s=
`.46888/.47040/.47138`，三档最长67帧且0 OOM/nonfinite，按封存规则选择B32；下一有信息量证据只能来自formal
macro5后的single-checkpoint strict paired400。

CVEG formal把最早失败再推进了一层。canonical fresh retry在macro3 sealed feasible-set gate拒绝且无checkpoint；
相同合同诊断复现的macro1逐值一致，macro2 functional loss、blind negative ratio与persisted bank也一致，但
task23的一条paired rollout从`candidate gain`翻成`candidate loss`，于是它从beneficial变为harmful，current hard
equality guard改变，shared Program随即分叉。诊断run因此完成5宏，而canonical run在第三步不可行。这个证据
不是“普通随机性可以忽略”：它说明只有两状态的binary candidate classification被直接放大成高维exact equality，
造成checkpoint路径依赖。固定batch1、物理rank或逐元素一致既低效，也不能解决真实closed-loop成功边界本身的
不连续性。

诊断五宏bank=`10/14/16/16/16`、energy=`.354/.375/.287/.274/.498`，K2 candidate净成功=
`-3/-1/-3/-3/0`，全部E/negative/guard closure仍健康。因此cross-video E与minimum correction没有被否定；被
否定的是当前`binary K2 outcome -> current hard equality guard`作为稳定共同积累机制。仍需对diagnostic macro5
做一次directional strict400，因为closed-loop absolute高于内部稳定性surrogate：低分会同时否定当前组合的性能
价值，高分只说明E应由下一稳定credit接口继承，不能把这个checkpoint冒充可重复winner。

directional strict400最终为`131/400`、breadth6、per-task=`1/2/47/30/0/35/16/0`。相对old134严格配对为
`113 retained/18 gained/21 lost`，相对NPCG135为`114/17/21`；Object3净退6，Long1仍以18次换手换来净增2。
因此hard E在当前blind-credit组合中也没有absolute或retention价值。连续guard路线从PICK-GC138、SKNC137、
NPCG135到CVEG131，在Program/negative/rank/energy均健康时持续走低，说明最早可行动失效接口已不再是constraint
composition，而是shared credit acquisition本身；后继不能继续给single-video blind cotangent加硬保护。

第一性原理后继因此选择PVJFC，而不是删一个guard的小修。每task的两条ordered action-hidden videos各自在完整
v6+Program+rank16 LoRA路径上、对同一B20及同一policy RNG求Program cotangent；两条correct及各自negative以
`1/2+1/2`进入一次weighted joint solve。它允许same-task不同video有不同的有用motion，不把cotangent、feature或
LoRA先平均；同时swap invariance和duplicate-view退化确保没有顺序偏置或task权重翻倍。该设计只检验continuous
paired-video credit是否比single-point blind proposal更接近跨video共同support。

PVJFC canonical实现已经原位替换CVEG。每个task只构造一次B20 batch与keyed policy seed，两条view串行完成
独立Writer图、functional LoRA VJP和Program cotangent后立即释放大graph；dynamic queue收齐train24后只gather
48 correct、48 matched negative和48 cotangent，以固定半权做一个96-row ridge write。部署仍是一条video生成一套
完整rank16 LoRA；checkpoint只有Program memory与rank RNG。旧candidate guard、success bank、outcome rollout、
hard E和其active tests已删除。swap invariance、duplicate-view对独立闭式single-view ridge退化、world1--6
padded gather、信息墙与exact-resume的完整CPU回归为`329 passed`。

clean`a4f4c75`、gpu02:0--5 world6的唯一PVJFC macro0 profile把最早失败定位在condition shared-credit几何，
而不是paired cotangent内容或吞吐。positive/full rank=`48/96`，24/24 tasks的primary与companion derivative均负，
四suite joint derivative均负；negative/correct motion ratio=`.07266`且reversed/shuffled/wrong分别`14/15/16`
views达`.15`门，Program write、LoRA A/B、effective BA、固定action全非零。Phase A/total仅`49.918/51.543s`，
相比CVEG显著更快。唯一false是regularized condition=`597.861>200`；因此profile总门non-pass、无checkpoint且
formal不开放。这个结果淘汰的是unit-normalized PICK-GC raw condition rows与`.01` full96 ridge的当前组合，不能
外推否定continuous paired-video credit；同时也不授权调damping或放condition门。下一单变量应让condition表示中
承载paired useful cotangent的方向与跨video/static共同高能方向结构性分离，而不是继续堆guard或追漂亮rank。

selected macro0 correct48的cache谱进一步定位：primary/companion各自condition约`17.47/16.05`，合并后因
same-task pair cosine均值`.89895`升到`250.11`；pair-symmetric能量约`94.95%`，低能private方向只贡献形式rank。
reversed与correct又近反平行，Gram中反号不会抵消。CGIK-JC因此只把key从独立goal/causal改为
`[causal, goal*causal]`，让所有地址必须含有序过程并打破correct/reverse同一直线；cache预验condition=`163.88`、
rank48、same-task cosine`.85312`。这只授权一次真实full96 profile；历史PICK-GC152.61但strict138已经证明
condition改善不能选择模型。

CGIK-JC随后完成了真正raw-frame full96裁决。clean`623505b`、gpu02:0--5 world6 profile的rank仍为`48/96`，
24/24 tasks两个view和四suite均下降，negative ratio降到`.02457`且reversed/shuffled/wrong各`16/16`，完整
Program→LoRA→BA→action链与`50.032s` wall通过；但condition仅降到`270.188>200`，因此总门仍non-pass且没有
checkpoint。它证明causal-goal interaction确实打破了大部分correct/reverse共线与negative leakage，却没有清除
full96剩余shared mode。该结果只淘汰当前`[causal, goal*causal]`固定key×full96 ridge，不否定interaction或paired
continuous credit；也不授权调damping、门、block weight、seed、width、view数、rank、scale或dtype。

剩余谱分块把最早接口进一步定位：CGIK独立raw-causal块correct48 condition=`322.87`，加16条reversed后
`679.39`，reverse约98%能量落入correct causal span；signed interaction块对应`106.24→201.58`。因此MGCI-JC
只把第一块改成`abs(goal)*causal`，第二块仍为`goal*causal`。同一cache预验correct48=`108.81`、
correct+reverse16=`130.78`、same-task cosine mean/median=`.806/.848`。这不是新时序阶数或whitening，只是移除
已定位的独立raw-causal公共地址；只有一次真实full96 profile能授权formal。

MGCI-JC已原位实现：公开condition owner保持不变，唯一算法diff是`[u_c,u_g*u_c]→[abs(u_g)*u_c,u_g*u_c]`；
旧CGIK config继续sealed non-pass，MGCI使用fresh config/run/profile/completion/checkpoint/RNG/inspection和eval v11
identity。其余shared full96 solve、B20、negative schedule、optimizer与deployment合同逐节继承CGIK。显式LIBERO
assets root下完整CPU为`339 passed`，compileall、canonical config load与diff检查通过；这些只授权一次live profile。

clean`eb1e53b`唯一raw-frame full96 profile随后以12/12通过：condition=`174.813`，比PVJFC/CGIK的
`597.861/270.188`继续下降；positive/full rank=`48/96`，24/24双view和四suite derivative为负，negative motion
ratio=`.02088`且reversed/shuffled/wrong各16/16，Program→A/B→BA→fixed action全非零，wall=`49.841s`，OOM/
nonfinite/outcome rollout/negative policy forward均0。root没有checkpoint。MGCI因此只获得fresh`0→5`真实闭环
裁决资格；condition通过不是性能结论，也不授权profile重跑或小扫。

fresh formal`0→5`随后完成，五宏rank均`48/96`、negative/correct motion保持`.020--.030`且无OOM/nonfinite；
macro5 strict correct400=`134/400`、breadth6、per-task=`1/5/46/30/0/34/18/0`。old134→MGCI严格配对为
`114 retained/20 gained/20 lost/246 both-fail`、churn40；suite净值=`+1/-6/-1/+6`。held effective-BA相对old134
median cosine=`.99999719`、norm ratio=`1.00001636`、relative L2=`.00242453`，gained rows也没有更大变化。
因此MGCI只修复condition key谱和negative suppression，没有把blind B20 credit变成新的held useful tangent；
MGCI-JC+paired blind B20组合终局退役，不resume、不补controls。当前没有active successor，暂停讨论。
