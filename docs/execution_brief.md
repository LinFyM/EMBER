# EMBER Execution Brief

更新时间：2026-08-14。本文只定义当前实验与持续迭代的执行语义；实时进度见`active_session_handoff.md`，稳定
owner原则见`current_owner_requirements.md`，历史负结果见`research_history.md`。

## 1. Latest completed experiment and active method

V6 Shared-Core Procedure-Set Bridge已完成macro25 K4 strict paired correct400：`139/400`、breadth6、per-task=
`1/4/46/34/0/36/18/0`、per-suite=`5/80/36/18`，top3占`116/139=83.45%`。相对K1 old134为
`118 retained / 21 gained / 16 lost`、净`+5`、churn37；相对matched post-compiler K4 130为`118/21/12`、
净`+9`。增益主要来自Long1净`+7`，Goal3/Long2仍为0；breadth低于7，按门终局non-pass。

该方法保留冻结v6-fast的language-axial evidence、Semantic
Core、有向Procedure、native compiler remainder、factor heads和rank16 topology，只把同一个约197k set层从完整
compiler之后前移到shared Core读出与最终Core/Procedure fusion之间：原生Core reader先联合读取无序Core union；
每条有序Procedure再以同一shared Core解释；Procedure-set置换不变聚合后，原生AdaLN/post-fusion/factor heads只
运行一次。完整合同见`action_forecast_writer_v6_shared_core_procedure_set_bridge_design.md`。

canonical实现已完成。64项定向门验证native compiler阶段化前后逐tensor相等、K1在任意set参数下严格等于native
v6、K>1集合换位不变而video内倒序敏感、只有197120个Procedure-Set参数获得梯度；全量CPU=`371 passed`。
真实gpu01 world6 full24 B20 profile的macro1/2=`26.011/24.249s`、peak reserved=`40.758GB`、K1--K4各6、最长
323帧且0 OOM/nonfinite；macro1→2 q/k delta非零，完整set credit已打开。clean detached `502618b` fresh
macro0→25也已完整结束：25/25 metrics、completion与checkpoint齐全，总耗时`662.730s`，loss first/last=
`.101182/.095655`，0 OOM/nonfinite。macro25 K4 B8/B16/B32 deployment profile分别为
`.223358/.223313/.223323 LoRA/s`，三者stable且峰值reserved约`13.01GB`，按最高吞吐锁B8。

matched去混淆把Procedure-Set output归零后，训练残差相对当前LoRA的effective-BA relative-L2 mean只有
`.000918`、task-mean只有`.000574`；无参数shared-Core union + Procedure mean相对K1则为`.039674/.016982`。
所以closed-loop净增主要由更早数据流产生，后端训练层几乎没有学到可用改写。下一单变量是把同一个集合算子
前移到语言token对齐的per-video Semantic Core上，后端Procedure只做无参数mean；其它底座、rank16、B20、动态K
和训练recipe保持不变。

该V6 Semantic-Core Set Bridge的预注册authority与canonical实现已完成：同预算197120参数set对语言对齐Core
生成shared correction，step0严格保留139路径的output-zero数据流；旧Procedure-Set executable schema已删除。
full CPU=`372 passed`。当前仅授权真实GPU机制与full24 B20 profile；profile seal前不启动formal训练。

## 2. Single changed variable and training semantics

- v6的language-conditioned evidence、Semantic Core、有向Procedure、native compiler remainder、rank16 topology
  和factor heads全部加载macro400并冻结；
- 已完成方法唯一新增/训练的是跨video Procedure-Set层；下一迭代把同预算训练层移动到Semantic Core并移除后端
  trainable set，不加memory、rank变化、negative、
  expert、reward或新LoRA mapper；
- 24 train tasks构成一个完整macro，task内B20 mean后24-task等权；
- 每macro K1/K2/K3/K4各6，各task每四个macro覆盖全部K；
- K条video同task、action-hidden、互不重复且与action episodes错开，每条video保留stride-5完整序列；
- source policy与v6底座trainable参数为0；K1严格保留、K2--K4提供Procedure-Set functional gradient；
- profile只裁决真实wall/显存/batch，训练loss不选择checkpoint。

## 3. Closed-loop adjudication

完成K1逐tensor等价、K轴置换不变、video内顺序敏感、gradient/freeze和full24 profile后，fresh macro0→25并做K4
strict paired correct400。K1复用严格等价的old134 paired基线；K4若没有明确超过134或breadth低于7即终止，不扫
K/LR/temperature/seed。若K4超过150，封存single-checkpoint结果并补K1--K4 scaling及correct/same/wrong/
shuffled/reversed/no-video controls；机制成功后再建立同架构fresh训练recipe。

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
