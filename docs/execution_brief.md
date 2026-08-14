# EMBER Execution Brief

更新时间：2026-08-14。本文只定义当前实验与持续迭代的执行语义；实时进度见`active_session_handoff.md`，稳定
owner原则见`current_owner_requirements.md`，历史负结果见`research_history.md`。

## 1. Latest completed experiment and active successor

V6 Shared-Core Ordered-Procedure Common-Value已完成macro25 K4 strict paired correct400：`139/400`、breadth6，
per-task=`1/2/46/32/0/36/22/0`、per-suite=`3/78/36/22`。相对matched Shared-Core139严格配对=
`120 retained / 19 gained / 19 lost`、churn38；suite net=`-2/-2/0/+4`，Long1净增完全由Spatial/Object损失
支付。400 LoRAs、60 jobs、400 rows、15 workers全部exit0；按`<140`且breadth`<7`双门终局，不resume、
不补controls、不扫参。

机制已接通但credit无效：Procedure correction=`.09601`，effective-BA mean/task-mean=`.01397/.01392`，q/k/output
全部训练并改变action；train-seen 8×10 output-zero严格反事实却为trained/zero=`64/64`、`4 gained/4 lost`。
所以失败不是held-only泛化，也不是写出太小，而是B20 functional credit在train和held on-policy都只造成换手。

active successor为`action_forecast_writer_v6_ordered_procedure_on_policy_preference_design.md`。架构、K4视频、
rank16、冻结v6与部署图全部不变；macro25只作短AS cold start，新阶段关闭target action入口，以train24四初始化
真实success/failure的LOO executed-prefix preference优化同一19.7万参数shared Writer。旧Reward-Credit的一次
sub-ULP Program写入不恢复；新gradient由Adam累积到FP32 Procedure参数，并必须在effective BA和strict400中
证明真实作用。

实现与真实smoke已完成：正确assets下full CPU=`395 passed`；task4为`1/4` mixed，Writer gradient=`.0008012`，
effective-BA/fixed-action response=`.0001815/.0055719`。首个world5 formal先因合法长task超过默认600秒collective
timeout失败，PG失败后最慢rank才报告OOM；无checkpoint，不构成科学结果。reward timeout已改30分钟，同时移除
CFM期间不必要的compiler graph；同一B8 task4全部科学量逐位不变，peak reserved=`40.712GB`、exit0。task0
homogeneous面板严格zero-CFM/zero-gradient。formal contract以clean `fa53ce4`重新seal；现在fresh启动full24
cycle1，不从失败root恢复，也不改变B8/Nmc4/样本或objective。

## 2. Active single changed variable and training semantics

- v6 evidence、shared Core、有向Procedure、Common-Value operator、native compiler与rank16全部沿用macro25；
- 只训练Procedure q/k/output；source policy与v6底座trainable参数为0；
- 每个train task由K4 videos生成一套LoRA，在四个random resets闭环执行；video仍action-hidden；
- reward阶段source/teacher/validation/test action reads为0，只保留当前policy真实executed prefixes；
- LOO binary advantage、episode/task等权、Nmc4 executed-prefix CFM，24 tasks后一次shared AdamW update；
- dynamic work queue只改变physical owner，task/video/env/policy/flow seeds不含rank；
- cycle1后立即strict paired400，reward objective和train80不选择checkpoint。

## 3. Closed-loop adjudication

Ordered-Procedure AS strict139/breadth6已触发终止门且没有resume或补controls。reward cycle1只认同schedule K4
strict paired400：`<144`、breadth<7、相对139 lost>10或gained不超过lost即终局；`144..150`且retention/三suite
趋势过门只允许cycle2；首次>=144才补因果controls，最终成功仍要求strict>150与健康controls。

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
