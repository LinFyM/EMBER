# Video-Conditioned Expert-Manifold: Historical Evidence

状态：2026-08-11归档摘要。本路线及其Reward后继均已完成裁决；本文不再是活动design，不包含可执行的下一步。
当前状态见`docs/active_session_handoff.md`，完整方法谱系见`docs/research_history.md`。原2923行设计与逐步证据可由
`git show 3a6f801:docs/action_forecast_writer_video_expert_manifold_design.md`读取。

## 1. Original hypothesis

此前Writer长期出现两类同时存在的问题：

- generated LoRA能量过低、近rank1、跨target/跨task过度共线，难以产生足够policy作用；
- 即使视频路径有非零hidden/LoRA/action传递，训练仍在task之间换手，correct顺序没有稳定带来有用方向。

Expert-Manifold路线提出：先为24个development-train tasks分别训练task-local SFT LoRA，得到确实能驱动
frozen policy的parameter targets；再让共享Writer从language + action-hidden video生成或组合这些policy-effective
更新。目标不是部署时检索专家，而是让专家定义训练域内“有用LoRA长什么样”。

部署信息墙始终要求：

```text
exact task language + exactly one action-hidden teacher video
    -> shared Writer/compiler
    -> one complete 38-target rank-16 LoRA
    -> frozen source policy
```

held部署不得读取expert bank、task ID、filename、teacher action/proprio/reward或第二套LoRA。

## 2. Task-expert bank

正式root：

`runs/outputs/pi05_task_expert_bank_formal_step1000_r6_81101fe_20260807`

最初从clean commit`81101fe`以6个独立workers训练24 tasks；每task保存step250/500/1000。最后50步24-task等权
mean action loss为`.115355/.107207/.105372`。后来所有experts沿同一root、同一合同统一续到1500/2000，
没有按task挑训练步数。

development-train direct-expert closed-loop：

| step | success / 1200 |
| ---: | ---: |
| 250 | 432 |
| 500 | 557 |
| 1000 | 624 |
| 1500 | 638 |
| 2000 | 658 |

step2000有23/24 tasks非零，task9仍为0；相对step1000 paired gains/losses=`77/57`。统一target因此选step2000，
但它只是privileged train-task target质量证据，不是Writer held结果。

full-bank step250/500/1000 geometry：effective-LoRA norm中位`2.792/3.652/4.170`，stable-rank中位
`1.126/1.129/1.129`，跨task effective cosine中位`.108/.095/.100`。16个rank coordinates均active，top4
coordinate energy约`.262/.260/.258`；q/v的B columns较coherent，action更分离。step1500/2000 norm约
`4.212`且几何基本饱和。

这些结果纠正了一个错误目标：正常SFT专家本身也可能低stable-rank、q-dominant和跨列coherent；不能把“均匀
高秩”机械定义为健康。真正重要的是这些更新对policy有效。

## 3. What task experts do and do not identify

experts解决：

- 给出task-local、policy-effective的完整LoRA target；
- 提供真实SFT能量、rank participation、target ownership与跨task方向参考；
- 允许定位失败发生在video representation、routing、compiler还是policy support。

experts不解决：

- held task泛化；
- same-task不同videos的特异性；
- correct相对shuffled/reversed的时间因果；
- shared checkpoint中的多task稳定共存。

根本原因是同一task的expert target`E_t`对所有videos恒定。一个Writer可以完美重建`E_t`，仍只靠language或静态
scene识别task；expert supervision本身没有要求解释视频中“先做什么、后做什么”。

## 4. Addressless topological Writer

第一版把action-hidden phase16×3072 frozen video features切成168个`[16,512]` topological chunks，通过
chunk/rank axial Writer直接重建完整expert factors。它能把generated LoRA norm推到`4.55`，接近expert
`4.21`，但stable-rank约`1.000001`、top singular energy约`.999999`，q/v/action B-column cosine约`.99999`；
nearest expert effective cosine仅`.00797`。

更关键的是train24自身demo0对own expert raw/effective cosine也只有`.0233/.0108`。所以最早失败发生在训练域
decoder topology：缺少明确target/rank地址后，不同输出槽共享同一方向。held strict correct=`48/400`，与
source identity相当。它不是“能量仍太小”的问题。

## 5. Topology-address binding

后继只增加静态chunk/rank地址，让video保持唯一dynamic value。早期macro3可得到norm`3.36`、stable-rank
`1.35`、own-expert cosine`.134`，显著优于addressless；但不同task generated effective cosine约`.869`，仍
过于公共。正式输出约norm`3.20`、stable-rank`1.32`，不同视频/任务的LoRA几乎同向，strict correct=`75/400`。

结论：address能避免必然的topology identity collapse，却不足以使动态视频证据选择正确policy方向。

## 6. Causal barycentric reconstruction

该方案保留phase-centered causal video representation，用train24 expert作为basis，解析求每条video的24维
barycentric coefficients。它把“生成任意dense factors”改成“在已知有效更新的流形上选坐标”。

raw-factor线性组合存在`(sum B_i)(sum A_i)`的cross terms，因此不保持effective update；后来改为在effective
`BA`空间组合，再压回public rank16。one-hot reconstruction可达到target-level effective cosine约`.998`，
generated LoRA norm/stable-rank/top-energy也与experts同档。

然而closed-loop仍低：causal barycentric strict=`63/400`。400条输出的geometry健康，coordinate inversion也
可靠，但same-task/cross-task坐标仍过度相似，nearest expert support不足。它证明“位于SFT-like流形”不是
“位于held task的有用support”。

## 7. Policy-effective soft/hard expert bank

为分离mixture误差与support问题，路线又比较：

- soft policy-effective barycentric：held screen=`15/80`；
- hard nearest-expert route：held screen=`3/80`，且compiler可近精确复现被选expert。

hard arm把混合、rank压缩与重建误差几乎移除后反而更差，因此当前24个train experts不能直接作为held
deployment dictionary。禁止通过top-k、temperature、confidence、global scale、rank或few-shot平均继续救这条
bank route。

该结论只否定“当前reader + 当前24-expert held dictionary”。它不否定task-level parameter targets可用于训练，
也不否定未来从train geometry推导连续representation。

## 8. Transition to frozen-v6 and Reward credit

Expert-Manifold之后，研究转而保留历史v6的强macro0 policy geometry，只训练或计算小的condition-local
correction：whole-LoRA attraction、Expert-Component Projection、Tangent Tube、Expert-Flow Teacher、
Balanced residual、RLS reconciliation与Reward-Credit。

这条后继链的重要结果：

- whole-LoRA attraction径向收缩，曲线`134/127/105/123`；
- Expert-Component Projection为`134/133/120`，expert component提高但orthogonal drift更大；
- Tangent Tube控制半径却没有旋正方向，`134→131`；
- Expert-Flow teacher在matched真实flow loss仅`2/24` tasks、`0/4` suites优于baseline；
- Balanced residual达到`134/140/139`，但10→25仍`12 gained/13 lost`，不同video correction近正交；
- RLS row保留达到140却full400 lost15，说明offline support不等于held occupancy；
- Reward-Credit产生有内容的video-conditioned Program和continuous tangent，但cycle1仍134且`14/14`换手；
- q/v tangent约`1e-8 RMS`，低于native BF16 factor局部ULP，引出最终rank-reserved实验；
- uniform rank14 compiler-only 138但lost15，online128且lost21，证明compression与regeneration均损伤support。

最终rank14设计与裁决保留在`docs/action_forecast_writer_qv_rank_reserved_native_reward_design.md`。

## 9. Few-shot relation

K4历史提供了一个仍值得保留的动机：多条同task videos可通过集合聚合削弱单demo偶然性，同时保留每条视频
内部时序。它改善了permutation、same/leave-one-out内部稳定性，但best strict仅108、full24 retention约`.043`。

因此few-shot不应作为Expert-Manifold bank的平均补丁，也不能平均生成后的LoRA。未来若重开，应独立设计
ordered-per-video + permutation-invariant-across-videos的matched模型，固定`k`或显式cardinality mask，并与
one-shot按计算和paired controls对照。

## 10. Durable conclusions

1. task-local SFT LoRA是有用的policy target，但不是视频teacher。
2. LoRA energy、rank participation、expert cosine与重建精度都不是closed-loop充分条件。
3. raw-factor mixture会产生cross terms；任何流形组合都应在effective update层定义与验证。
4. held task需要的policy support不能由24个train experts的nearest/convex dictionary自动覆盖。
5. 视频route、same-video consistency与正确顺序敏感都必须继续传到effective BA、action和strict rollout。
6. shared Writer的核心难题是condition-specific credit与跨task/video稳定共存，不是单纯增加decoder容量。
7. 新路线应选择性复用有效子机制，不得恢复整条已退役Expert-Manifold执行路径。
