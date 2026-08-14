# EMBER Execution Brief

更新时间：2026-08-14。本文只定义当前实验与持续迭代的执行语义；实时进度见`active_session_handoff.md`，稳定
owner原则见`current_owner_requirements.md`，历史负结果见`research_history.md`。

## 1. Latest completed experiment and active successor

V6 Ordered-Procedure On-Policy Preference已完成一次fresh reward cycle和K4 strict paired correct400：`138/400`、
breadth7、per-task=`2/5/46/33/0/36/15/1`、per-suite=`7/79/36/16`。相对同schedule AS139严格配对=
`120 retained / 18 gained / 19 lost`、churn37；Spatial净`+4`，Object净`+1`，Goal净0，Long净`-6`，其中
Long1=`3 gained / 10 lost`。按`<144`、lost`>10`且gained不超过lost三项门终局，不运行cycle2、不补controls、
不扫参。

这不是reward无效或写出断路：cycle1的24 tasks/96 rollouts得到64 successes、14 mixed tasks和四suite完整credit，
q/k/output均更新，5个deployment probes的BA/action响应非零；AS→reward effective-BA mean relative-L2仅
`.003323`、cosine`.999995`、norm ratio`1.000762`，却翻转37条闭环结果。最早失效接口是同一个shared Adam
candidate合并异质task方向后没有保住已有support，而不是幅度、rank、LoRA健康度、Procedure或compiler。

active successor为`action_forecast_writer_v6_ordered_procedure_final_shared_support_projection_design.md`。它只改变
最终shared update：保留同一K4视频输入、rank16 V6 Writer、19.7万FP32参数、LOO reward proposal和完整部署图；
对task汇合后的actual parameter delta施加train24成功executed-prefix loss的一阶非增约束。它不是继续cycle2，
也不是恢复旧Program bank/guard路线。实现、fresh schema、完整CPU=`400 passed`与task4真实mixed smoke已经完成；
smoke保持raw BA/action response逐位一致，support路径非零，cycle wall仅为raw的`1.077x`且peak reserved=
`36.774GB`。config已seal，下一步从clean pushed commit正式fresh full24 cycle。

## 2. Active single changed variable and training semantics

- v6 evidence、shared Core、有向Procedure、Common-Value operator、native compiler与rank16全部沿用macro25；
- 只训练Procedure q/k/output；source policy与v6底座trainable参数为0；
- 每个train task由K4 videos生成一套LoRA，在四个random resets闭环执行；video仍action-hidden；
- reward阶段source/teacher/validation/test action reads为0，只保留当前policy真实executed prefixes；
- LOO binary advantage、episode/task等权、Nmc4 executed-prefix CFM仍形成同一个raw shared AdamW candidate；
- 每个至少一条成功rollout的train task额外形成一个task-equal success-support tangent；只有最终actual parameter
  delta被投影到全部support half-spaces，video/representation/compiler/optimizer proposal均不变；
- dynamic work queue只改变physical owner，task/video/env/policy/flow seeds不含rank；
- fresh constrained cycle通过机制门后立即strict paired400，reward objective和train80不选择checkpoint。

## 3. Closed-loop adjudication

Ordered-Procedure AS139与raw reward138均已终局且不得resume。新的support-constrained cycle仍只认同schedule K4
strict paired400：`<144`、breadth<7、相对139 lost>10或gained不超过lost即终局；首次达到`144..150`且
retention/三suite趋势过门，才讨论同合同续训；首次>=144才补因果controls，最终成功仍要求strict>150与健康controls。

报告aggregate、8项per-task、4 suite totals、breadth、retained/gained/lost、top-task concentration和K1→K4
success-set变化。不能用K1/K4 union、LoRA norm或functional loss冒充同一condition的能力。

## 4. Continuous adjudication loop

每轮strict结果完成后，按以下顺序分析：

1. absolute、per-task/per-suite、breadth；
2. 相对最接近方法和历史强基线的retained/gained/lost与能力集中；
3. 若有相邻checkpoint，分析persistent/gained/lost与union gap；
4. 沿`input evidence -> Core/Procedure -> set/compiler -> effective BA -> fixed action -> rollout`定位最早失效接口；
5. 分离科学non-pass与明确工程合同违约；
6. 只对最早接口提出一个主要因果变量，写可证伪authority后实现；
7. 做最小必要CPU/机制验证和吞吐profile，尽快回到真实paired400。

不得用loss、cosine、rank或漂亮五臂margin代替absolute，也不得通过rank/scale/seed/dtype/temperature小扫救一个
失败checkpoint。owner的局部建议不能导致整套已认可方案无证据重写。

## 5. GPU and efficiency

- launch前同时live检查gpu01/gpu02；单节点至多6张，有多少真正合适就用多少；
- 允许在显存峰值余量充足、低util且不明显干扰他人的卡上共驻；不抢占、kill、reset或dummy占卡；
- evaluator当前保守共驻门为util≤10%、已用≤8GiB且剩余≥32GiB；任一越界即拒绝；
- 多卡训练设置`NCCL_P2P_DISABLE=1`、GPU-local NUMA和deferred NCCL；
- fresh可用world1--6；exact resume锁原world topology；
- evaluator用动态cost queue和persistent model/env，不静态拆task；
- 以真实samples/s、LoRA/s、最长视频稳定性和显存峰值选择batch；
- 接受正常BF16/TF32低位差异，不重复forward、固定batch1、扩dtype、逐tensor scan或增加hash。

## 6. Storage, Git and artifacts

- 大run前查询`strg01`上的独立user quota，测canonical root并估计checkpoint/cache/temp峰值；
- formal训练与评测来自clean pushed commit的detached worktree；
- 新run使用fresh output root，不覆盖或部分复用中止/不兼容root；
- formal保留run contract、metrics、macro checkpoints、completion、400 raw rows、aggregate和decision analysis；
- profile/smoke roots只作机制/吞吐证据，不冒充formal；
- meaningful结果更新current state、execution brief、current design、task plan、findings和research history；
- 不把历史命令重新复制进多个文档，精确命令以run contract/invocations为准。

## 7. Collaboration boundary

owner授权在核心目标、信息墙与效率原则内持续自主迭代。当前暂不使用subagents。只有真正需要改变核心目标、
扩大权限、处理破坏性操作或遇到无法从本地证据解决的关键歧义时才向owner停下来请求决定。
