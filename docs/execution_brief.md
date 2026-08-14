# EMBER Execution Brief

更新时间：2026-08-15。本文只定义当前实验与持续迭代的执行语义；实时进度见`active_session_handoff.md`，稳定
owner原则见`current_owner_requirements.md`，历史负结果见`research_history.md`。

## 1. Latest completed experiment and next decision boundary

V6 Layerwise Action-Probe Conditioned Procedure Reader（V6-LPCP）已完成fresh macro0->25和K4 strict paired
correct400：`143/400`、breadth7、per-task=`1/4/48/35/0/38/16/1`、per-suite=`5/83/38/17`、top3=
`121/143=.84615`。相对同schedule AS139严格配对=`120 retained / 23 gained / 19 lost / 238 both-fail`、
churn42、net`+4`、p=`.643969`；suite net=`+2/+5/+2/-5`。它count-only追平不同teacher schedule的历史
v6-fast143并把breadth从6提到7，但按`<144`和lost`>10`两项门终局，不resume50、不补controls或参数小扫。

这不是carrier没有运行：18层native Action-probe states在同一次真实context forward中旁读，倒序使
query-delta/Program relative-L2=`2.0572/.40414`，constant query-delta max-abs=`3.38e-8`；训练后query
projection norm=`.142632`，reader/controller均有非零更新。部署B32=`.221500 LoRA/s`，相对AS139仅慢约1.6%。
但全400 effective-BA只相对AS139移动mean/median `.002653/.001916`，cosine=`.99999479`、norm ratio=
`.99997391`，LoRA norm/rank/q-v-action能量结构几乎不变。

first4同task correction coherence mean/median=`.61786/.56804`，所以本轮没有复现“同task视频修正近正交”；
更直接的反例是Goal3：BA改写`.004224`、coherence`.88373`仍`0/50`，而Long1只改`.001324`却
`7 gains/13 losses`。train24 functional first5/last5仅`.098880/.097109`，14 tasks改善、10 tasks变差。
最早失效接口因此是**conditioned Procedure经冻结fusion/compiler承诺成非常小的AS139邻域方向，blind B20
functional credit又不能判断该方向是否覆盖held on-policy occupancy**。carrier本身已经通过门，故不直接触发
literal-memory替换。

当前已选择policy-aligned credit分支：同schedule AS139/LPCP的严格成功集合union=`162`，说明超过150所需support
已分别存在于reference与candidate，但不是一个可部署checkpoint。PCSD用train24同初态K2两臂，只选择唯一成功
arm的executed trajectory，在当前candidate LoRA下做positive CFM distillation；ties为zero，不对失败轨迹做
anti-imitation。部署架构、K4输入、rank16、reader/conditioner和frozen compiler不变，只训练65,536参数
`query_delta.weight`。authority=
`docs/action_forecast_writer_v6_lpcp_paired_causal_success_distillation_design.md`。

## 2. Completed changed variable and training semantics

- 冻结AS139的Semantic Core、有向Procedure Value、K-set、fusion/compiler、38-target rank16 FactorHeads；
- 同一次真实图像+语言+50 Action-probe joint forward旁读18层hidden，不增加第二次backbone forward；
- shared rank queries逐层读取，每video一次causal controller，再生成layer/rank-aligned Procedure Query delta；
- 只训练reader/controller/zero-init query projection，source policy与AS139底座trainable参数为0；
- train24 task-complete、B20同task跨episodefunctional queries、K1--K4每macro各6，其余objective/recipe不变；
- Writer仍由exact language和action-hidden videos一次生成完整LoRA，rollout期间不反复观看视频。

## 3. Closed-loop adjudication

Ordered-Procedure AS139、raw reward138、ADSP138与V6-LPCP AS阶段均已终局且不得resume或小扫。PCSD是
fresh-incompatible reward-calibration stage，不是LPCP macro50续训；canonical实现与CPU门已完成，下一步做
full24 paired cycle，只有机制门
通过才立即strict400。`>150`仍是性能追求，但约145若要成立，必须由相邻checkpoint低churn、same-task-video
鲁棒和健康video controls共同认证；单点高分不能作为方法结论。

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
