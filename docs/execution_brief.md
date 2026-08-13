# EMBER Execution Brief

更新时间：2026-08-14。本文只定义当前实验与持续迭代的执行语义；实时进度见`active_session_handoff.md`，稳定
owner原则见`current_owner_requirements.md`，历史负结果见`research_history.md`。

## 1. Latest completed experiment and active method

Dynamic-K Task-Grounded Full-Factor Rank-8 Writer已完成formal fresh macro0→50及K1 strict paired400。结果为
`91/400`、breadth5、per-task=`4/1/38/0/0/37/11/0`、per-suite=`5/38/37/11`，top3占`86/91`。相对
matched fixed-A macro50 88为`70 retained/21 gained/18 lost`，仅净增3；低于fixed-A best96、Direct-B102、
old134、compiler138、online128和v6-fast143。按预注册125门终局non-pass，不resume、不补五臂。

matched first4 states/task中，Full-Factor相对fixed-A的raw A cosine/norm ratio=`.735154/1.376207`，raw B=
`.248553/.062232`，effective BA=`.058529/.244792`。训练loss几乎相同却形成larger-A/tiny-B的弱、近正交policy
update，说明当前B20 surrogate不能约束有用factor allocation。该证据终止当前rank8前端/mapper小修。

当前active方法是V6 Dynamic Slot-Set Bridge：每条video独立运行冻结原生v6 evidence→Core→ordered Procedure→
320 policy slots；每个对应slot只沿K轴做置换不变的mean backbone + selected centered residual；原生v6 factor heads
只解码一次。K=1中心化残差恒零，因此严格等于历史v6；K>1才学习same-task共同程序。首轮只训练约197k Slot-Set
参数，warm start只作机制开发，若成功仍需从零训练。完整合同见
`action_forecast_writer_v6_dynamic_slot_set_bridge_design.md`。

canonical实现与GPU机制门已通过：K1的76个LoRA tensors逐元素等于native v6，只有197120个Slot-Set参数可训练，
v6/source无梯度，真实video倒序明显改变Program；全量CPU=`370 passed`。K2/K4换位只产生BF16 batched-forward
低位差异，不为此拆分forward或降低吞吐。当前待完成full24 B20 profile并seal正式训练合同。

## 2. Single changed variable and training semantics

- v6的language-conditioned evidence、Semantic Core、有向Procedure、compiler、rank16 topology和factor heads全部
  加载macro400并冻结；
- 唯一新增/训练的是跨video Slot-Set层，不加memory、rank变化、negative、expert、reward或新LoRA mapper；
- 24 train tasks构成一个完整macro，task内B20 mean后24-task等权；
- 每macro K1/K2/K3/K4各6，各task每四个macro覆盖全部K；
- K条video同task、action-hidden、互不重复且与action episodes错开，每条video保留stride-5完整序列；
- source policy与v6底座trainable参数为0；K1严格保留、K2--K4提供Slot-Set functional gradient；
- profile只裁决真实wall/显存/batch，训练loss不选择checkpoint。

## 3. Closed-loop adjudication

完成K1逐tensor等价、K轴置换不变、video内顺序敏感、gradient/freeze和full24 profile后，首个正式节点为macro25
K4 strict paired correct400。K1复用严格等价的old134 paired基线；K4若没有明确超过134或breadth低于7即终止，
不扫K/LR/temperature/seed。若K4超过150，封存single-checkpoint结果并补K1--K4 scaling及correct/same/wrong/
shuffled/reversed/no-video controls；机制成功后再授权同架构fresh训练。

报告aggregate、8项per-task、4 suite totals、breadth、retained/gained/lost、top-task concentration和K1→K4
success-set变化。不能用K1/K4 union、LoRA norm或functional loss冒充同一condition的能力。

## 4. Continuous adjudication loop

每轮strict结果完成后，按以下顺序分析：

1. absolute、per-task/per-suite、breadth；
2. 相对最接近方法和历史强基线的retained/gained/lost与能力集中；
3. 若有相邻checkpoint，分析persistent/gained/lost与union gap；
4. 沿`input evidence -> per-video Program -> set -> M2P -> mapper -> effective BA -> fixed action -> rollout`定位
   最早失效接口；
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
